#!/usr/bin/env python3
"""Align GO, Reactome, STRING and CollecTRI to the real global gene vocabulary."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import heapq
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from vcc_prior_model.vocabulary import GeneVocabularyArtifacts

WORKSPACE = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def split_field(value: str) -> list[str]:
    return [item.strip() for item in value.replace('"', "").split("|") if item.strip()]


class GeneResolver:
    """Resolve approved HGNC symbols and unambiguous historical aliases."""

    def __init__(self, hgnc_path: Path, vocabulary: GeneVocabularyArtifacts) -> None:
        self.global_by_symbol = vocabulary.global_index_by_symbol
        alias_candidates: dict[str, set[str]] = defaultdict(set)
        self.ensembl_to_symbol: dict[str, str] = {}
        self.uniprot_to_symbol: dict[str, str] = {}
        with hgnc_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                approved = row["symbol"].strip()
                if approved not in self.global_by_symbol:
                    continue
                aliases = [
                    approved,
                    *split_field(row.get("alias_symbol", "")),
                    *split_field(row.get("prev_symbol", "")),
                ]
                for alias in aliases:
                    alias_candidates[alias].add(approved)
                for ensembl in split_field(row.get("ensembl_gene_id", "")):
                    self.ensembl_to_symbol[ensembl.split(".")[0]] = approved
                for uniprot in split_field(row.get("uniprot_ids", "")):
                    self.uniprot_to_symbol[uniprot] = approved
        self.alias_to_symbol = {
            alias: next(iter(symbols))
            for alias, symbols in alias_candidates.items()
            if len(symbols) == 1
        }

    def symbol(self, value: str) -> str | None:
        value = value.strip()
        if value in self.global_by_symbol:
            return value
        return self.alias_to_symbol.get(value)

    def ensembl(self, value: str) -> str | None:
        return self.ensembl_to_symbol.get(value.strip().split(".")[0])

    def uniprot(self, value: str) -> str | None:
        return self.uniprot_to_symbol.get(value.strip())

    def index(self, value: str, *, kind: str = "symbol") -> int | None:
        resolver = getattr(self, kind)
        symbol = resolver(value)
        return None if symbol is None else self.global_by_symbol[symbol]


def save_memberships(
    output: Path, name: str, pairs: set[tuple[int, str]]
) -> tuple[np.ndarray, list[str]]:
    terms = sorted({term for _, term in pairs})
    term_index = {term: index for index, term in enumerate(terms)}
    ordered = sorted(pairs, key=lambda pair: (pair[0], term_index[pair[1]]))
    gene = np.asarray([pair[0] for pair in ordered], dtype=np.int64)
    term = np.asarray([term_index[pair[1]] for pair in ordered], dtype=np.int64)
    np.save(output / f"{name}_gene_index.npy", gene, allow_pickle=False)
    np.save(output / f"{name}_term_index.npy", term, allow_pickle=False)
    with (output / f"{name}_terms.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["term_index", "term_id"])
        writer.writerows(enumerate(terms))
    return gene, terms


def prepare_go(path: Path, resolver: GeneResolver) -> set[tuple[int, str]]:
    pairs: set[tuple[int, str]] = set()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line or line.startswith("!"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 11 or "NOT" in fields[3].split("|"):
                continue
            candidates = [fields[2], *fields[10].split("|")]
            gene_index = None
            for candidate in candidates:
                gene_index = resolver.index(candidate)
                if gene_index is not None:
                    break
            if gene_index is not None:
                pairs.add((gene_index, fields[4]))
    return pairs


def prepare_reactome(path: Path, resolver: GeneResolver) -> set[tuple[int, str]]:
    pairs: set[tuple[int, str]] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 2 or not fields[1].startswith("R-HSA-"):
                continue
            identifier = fields[0]
            identifier_kind = "ensembl" if identifier.startswith("ENS") else "uniprot"
            gene_index = resolver.index(identifier, kind=identifier_kind)
            if gene_index is not None:
                pairs.add((gene_index, fields[1]))
    return pairs


def prepare_string(
    info_path: Path,
    links_path: Path,
    resolver: GeneResolver,
    *,
    minimum_score: int,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    protein_to_gene: dict[str, int] = {}
    with gzip.open(info_path, "rt", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            protein = row.get("#string_protein_id") or row.get("string_protein_id")
            symbol = row.get("preferred_name")
            if not protein or not symbol:
                continue
            gene_index = resolver.index(symbol)
            if gene_index is not None:
                protein_to_gene[protein] = gene_index

    neighbours: dict[int, list[tuple[int, int]]] = defaultdict(list)
    with gzip.open(links_path, "rt", encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            protein_a, protein_b, score_text = line.split()
            score = int(score_text)
            if score < minimum_score:
                continue
            gene_a = protein_to_gene.get(protein_a)
            gene_b = protein_to_gene.get(protein_b)
            if gene_a is None or gene_b is None or gene_a == gene_b:
                continue
            for source, destination in ((gene_a, gene_b), (gene_b, gene_a)):
                heap = neighbours[source]
                item = (score, destination)
                if len(heap) < top_k:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)

    retained: dict[tuple[int, int], int] = {}
    for source, heap in neighbours.items():
        for score, destination in heap:
            edge = (min(source, destination), max(source, destination))
            retained[edge] = max(score, retained.get(edge, 0))
    ordered = sorted(retained.items())
    edge_index = np.asarray([edge for edge, _ in ordered], dtype=np.int64).T
    weight = np.asarray([score / 1000.0 for _, score in ordered], dtype=np.float32)
    return edge_index, weight


def truthy(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "t"}


def first_present(row: dict[str, str], names: tuple[str, ...]) -> str | None:
    return next((row[name] for name in names if name in row and row[name]), None)


def resolve_interaction_gene(value: str, resolver: GeneResolver) -> int | None:
    result = resolver.index(value)
    if result is not None:
        return result
    return resolver.index(value, kind="uniprot")


def prepare_collectri(
    path: Path, resolver: GeneResolver
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    retained: dict[tuple[int, int], tuple[int, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            source_text = first_present(
                row, ("source_genesymbol", "source_symbol", "source")
            )
            target_text = first_present(
                row, ("target_genesymbol", "target_symbol", "target")
            )
            if not source_text or not target_text:
                continue
            source = resolve_interaction_gene(source_text, resolver)
            target = resolve_interaction_gene(target_text, resolver)
            if source is None or target is None or source == target:
                continue
            stimulation = truthy(
                first_present(row, ("consensus_stimulation", "is_stimulation"))
            )
            inhibition = truthy(
                first_present(row, ("consensus_inhibition", "is_inhibition"))
            )
            if stimulation == inhibition:
                continue
            sign = 1 if stimulation else -1
            references = first_present(row, ("references", "references_stripped")) or ""
            confidence = float(max(1, len({item for item in references.split(";") if item})))
            edge = (source, target)
            previous = retained.get(edge)
            if previous is None or confidence > previous[1]:
                retained[edge] = (sign, confidence)
    ordered = sorted(retained.items())
    edge_index = np.asarray([edge for edge, _ in ordered], dtype=np.int64).T
    sign = np.asarray([value[0] for _, value in ordered], dtype=np.int8)
    confidence = np.asarray([value[1] for _, value in ordered], dtype=np.float32)
    confidence = np.log1p(confidence) / np.log1p(confidence.max(initial=1.0))
    return edge_index, sign, confidence.astype(np.float32, copy=False)


def degree(
    num_genes: int, edge_index: np.ndarray, *, directed: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    outgoing = np.bincount(edge_index[0], minlength=num_genes)
    incoming = np.bincount(edge_index[1], minlength=num_genes)
    if directed:
        return outgoing, incoming
    combined = outgoing + incoming
    return combined, combined


def coverage(counts: np.ndarray, selection: np.ndarray) -> dict[str, Any]:
    selected = counts[selection]
    return {
        "covered_genes": int((selected > 0).sum()),
        "total_genes": int(selected.size),
        "fraction": float((selected > 0).mean()) if selected.size else 0.0,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    vocabulary = GeneVocabularyArtifacts(args.vocabulary)
    resolver = GeneResolver(args.hgnc, vocabulary)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    go_pairs = prepare_go(args.go, resolver)
    reactome_pairs = prepare_reactome(args.reactome, resolver)
    go_gene, go_terms = save_memberships(output, "go", go_pairs)
    reactome_gene, reactome_terms = save_memberships(output, "reactome", reactome_pairs)
    string_edges, string_weight = prepare_string(
        args.string_info,
        args.string_links,
        resolver,
        minimum_score=args.string_minimum_score,
        top_k=args.string_top_k,
    )
    grn_edges, grn_sign, grn_weight = prepare_collectri(args.collectri, resolver)
    np.save(output / "string_edge_index.npy", string_edges, allow_pickle=False)
    np.save(output / "string_edge_weight.npy", string_weight, allow_pickle=False)
    np.save(output / "grn_edge_index.npy", grn_edges, allow_pickle=False)
    np.save(output / "grn_edge_sign.npy", grn_sign, allow_pickle=False)
    np.save(output / "grn_edge_weight.npy", grn_weight, allow_pickle=False)

    num_genes = vocabulary.num_genes
    go_count = np.bincount(go_gene, minlength=num_genes)
    reactome_count = np.bincount(reactome_gene, minlength=num_genes)
    string_degree, _ = degree(num_genes, string_edges)
    grn_out, grn_in = degree(num_genes, grn_edges, directed=True)
    output_selection = vocabulary.output_gene_index.numpy()
    output_index_set = set(output_selection.tolist())
    with (vocabulary.directory / "perturbation_target_mapping.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        target_selection = np.asarray(
            [int(row["global_index"]) for row in csv.DictReader(handle)],
            dtype=np.int64,
        )
    rows = []
    for index, symbol in enumerate(vocabulary.global_gene_symbols):
        rows.append(
            {
                "global_index": index,
                "gene_symbol": symbol,
                "in_vcc_output": int(index in output_index_set),
                "go_terms": int(go_count[index]),
                "reactome_pathways": int(reactome_count[index]),
                "string_degree": int(string_degree[index]),
                "grn_out_degree": int(grn_out[index]),
                "grn_in_degree": int(grn_in[index]),
                "active_prior_count": int(go_count[index] > 0)
                + int(reactome_count[index] > 0)
                + int(string_degree[index] > 0)
                + int(grn_out[index] + grn_in[index] > 0),
            }
        )
    with (output / "gene_prior_coverage.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    sources = {
        "hgnc": args.hgnc,
        "go": args.go,
        "reactome": args.reactome,
        "string_info": args.string_info,
        "string_links": args.string_links,
        "collectri": args.collectri,
    }
    all_prior = go_count + reactome_count + string_degree + grn_out + grn_in
    manifest = {
        "schema_version": 1,
        "global_gene_count": num_genes,
        "global_vocabulary_sha256": vocabulary.manifest["global_vocabulary"]["symbols_sha256"],
        "parameters": {
            "string_minimum_score": args.string_minimum_score,
            "string_top_k": args.string_top_k,
        },
        "go": {
            "terms": len(go_terms),
            "memberships": len(go_gene),
            "global": coverage(go_count, np.arange(num_genes)),
            "vcc": coverage(go_count, output_selection),
            "vcc_targets": coverage(go_count, target_selection),
        },
        "reactome": {
            "pathways": len(reactome_terms),
            "memberships": len(reactome_gene),
            "global": coverage(reactome_count, np.arange(num_genes)),
            "vcc": coverage(reactome_count, output_selection),
            "vcc_targets": coverage(reactome_count, target_selection),
        },
        "string": {
            "edges": int(string_edges.shape[1]),
            "global": coverage(string_degree, np.arange(num_genes)),
            "vcc": coverage(string_degree, output_selection),
            "vcc_targets": coverage(string_degree, target_selection),
        },
        "collectri": {
            "edges": int(grn_edges.shape[1]),
            "positive_edges": int((grn_sign > 0).sum()),
            "negative_edges": int((grn_sign < 0).sum()),
            "global": coverage(grn_out + grn_in, np.arange(num_genes)),
            "vcc": coverage(grn_out + grn_in, output_selection),
            "vcc_targets": coverage(grn_out + grn_in, target_selection),
        },
        "any_prior": {
            "global": coverage(all_prior, np.arange(num_genes)),
            "vcc": coverage(all_prior, output_selection),
            "vcc_targets": coverage(all_prior, target_selection),
            "vcc_active_source_histogram": {
                str(count): int(
                    np.sum(
                        (
                            (go_count[output_selection] > 0).astype(np.int8)
                            + (reactome_count[output_selection] > 0).astype(np.int8)
                            + (string_degree[output_selection] > 0).astype(np.int8)
                            + (
                                grn_out[output_selection] + grn_in[output_selection] > 0
                            ).astype(np.int8)
                        )
                        == count
                    )
                )
                for count in range(5)
            },
        },
        "sources": {
            name: {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for name, path in sources.items()
        },
    }
    with (output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vocabulary",
        type=Path,
        default=WORKSPACE / "priorcell/artifacts/gene_vocabulary",
    )
    parser.add_argument(
        "--hgnc",
        type=Path,
        default=WORKSPACE / "data/priors/hgnc_complete_set.txt",
    )
    parser.add_argument(
        "--go",
        type=Path,
        default=WORKSPACE / "data/priors/HUMAN-uniprot.gaf.gz",
    )
    parser.add_argument(
        "--reactome",
        type=Path,
        default=WORKSPACE / "data/priors/UniProt2Reactome.txt",
    )
    parser.add_argument(
        "--string-info",
        type=Path,
        default=WORKSPACE / "data/priors/9606.protein.info.v12.0.txt.gz",
    )
    parser.add_argument(
        "--string-links",
        type=Path,
        default=WORKSPACE / "data/priors/9606.protein.links.v12.0.txt.gz",
    )
    parser.add_argument(
        "--collectri", type=Path, default=WORKSPACE / "data/priors/collectri_9606.tsv"
    )
    parser.add_argument(
        "--output", type=Path, default=WORKSPACE / "priorcell/artifacts/priors"
    )
    parser.add_argument("--string-minimum-score", type=int, default=400)
    parser.add_argument("--string-top-k", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), indent=2))
