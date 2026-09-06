#!/usr/bin/env bash
set -euo pipefail

workspace="${VCC_WORKSPACE:-/sde/vcc/vcc2026/Maojianhan}"
state_dir="${workspace}/external/SE-600M"
jiang_dir="${workspace}/data/multicontext/jiang2025"
rpe1_dir="${workspace}/data/references/replogle_rpe1"
group="${1:-all}"

mkdir -p "${state_dir}" "${jiang_dir}" "${rpe1_dir}"

download() {
  local output_dir="$1"
  local output_name="$2"
  local expected_md5="$3"
  local url="$4"
  aria2c -c -x 8 -s 8 --file-allocation=none \
    --dir="${output_dir}" --out="${output_name}" "${url}"
  if [[ "${expected_md5}" != "-" ]]; then
    printf '%s  %s\n' "${expected_md5}" "${output_dir}/${output_name}" | md5sum --check -
  fi
}

download_hf() {
  local output_name="$1"
  local hub_url="https://huggingface.co/arcinstitute/SE-600M/resolve/main/${output_name}"
  local resolved_url
  resolved_url="$(curl -4 -sS -o /dev/null -w '%{redirect_url}' "${hub_url}")"
  if [[ -z "${resolved_url}" ]]; then
    resolved_url="${hub_url}"
  fi
  aria2c -c -x 8 -s 8 --file-allocation=none \
    --dir="${state_dir}" --out="${output_name}" "${resolved_url}"
}

download_state() {
  # Resolve Xet's signed CDN URL first; split requests against the Hub redirect
  # itself can stall behind a proxy.
  python3 "$(dirname "$0")/download_hf_range.py" \
    "https://huggingface.co/arcinstitute/SE-600M/resolve/main/se600m_epoch16.ckpt" \
    "${state_dir}/se600m_epoch16.ckpt" \
    --size 11548282659 \
    --sha256 b49bab144471f3b9318e1a661fb7b78bfa5110b500aee5c89d660f5f5927b7a5
  download_hf protein_embeddings.pt
  download_hf config.yaml
}

download_rpe1() {
  download "${rpe1_dir}" "ReplogleWeissman2022_rpe1.h5ad" \
    cc7f1ec50aeb3a3e1b4a6cfa713d80fa \
    "https://zenodo.org/api/records/10044268/files/ReplogleWeissman2022_rpe1.h5ad/content"
}

download_jiang() {
  # Independent Zenodo files are fetched concurrently; each individual file
  # still uses resumable ranges and is verified before the group completes.
  download "${jiang_dir}" "Seurat_object_IFNB_Perturb_seq.rds" \
    3eb5e7af1601bf562a5b20dea5de3dc9 \
    "https://zenodo.org/api/records/14518762/files/Seurat_object_IFNB_Perturb_seq.rds/content" &
  download "${jiang_dir}" "Seurat_object_IFNG_Perturb_seq.rds" \
    0fef1f14c36906e9c40e4d1c6aae6926 \
    "https://zenodo.org/api/records/14518762/files/Seurat_object_IFNG_Perturb_seq.rds/content" &
  download "${jiang_dir}" "Seurat_object_INS_Perturb_seq.rds" \
    c7b830dfcc020545c3f222cad5b13b34 \
    "https://zenodo.org/api/records/14518762/files/Seurat_object_INS_Perturb_seq.rds/content" &
  download "${jiang_dir}" "Seurat_object_TGFB_Perturb_seq.rds" \
    8e9b4d39a95ec5881a30be6a2df541d1 \
    "https://zenodo.org/api/records/14518762/files/Seurat_object_TGFB_Perturb_seq.rds/content" &
  download "${jiang_dir}" "Seurat_object_TNFA_Perturb_seq.rds" \
    60ed8bff6c749b1250f8fde9c5435c2e \
    "https://zenodo.org/api/records/14518762/files/Seurat_object_TNFA_Perturb_seq.rds/content" &
  download "${jiang_dir}" "DE_results_all_pathway.zip" \
    f077cba680a1affc599f5153d99b0e45 \
    "https://zenodo.org/api/records/14518762/files/DE_results_all_pathway.zip/content" &
  wait
  download "${jiang_dir}" "Pathway_genelist.rds" \
    f107354d6d075364a3c94e34f1ff5134 \
    "https://zenodo.org/api/records/14518762/files/Pathway_genelist.rds/content"
  download "${jiang_dir}" "A_readme.txt" \
    b05ff3d3117887faa4aaff1414030317 \
    "https://zenodo.org/api/records/14518762/files/A_readme.txt/content"
}

download_jiang_h5ad() {
  python3 "$(dirname "$0")/download_hf_range.py" \
    "https://huggingface.co/datasets/altoslabs/perturbench/resolve/main/jiang24_processed.h5ad.gz" \
    "${jiang_dir}/jiang24_processed.h5ad.gz" \
    --size 15232554616 \
    --sha256 dd890a8019b8a615010963b32e6e28ce42fd0b94a749602bc93efb2fa8af56cc
  python3 "$(dirname "$0")/download_hf_range.py" \
    "https://huggingface.co/datasets/altoslabs/perturbench/resolve/main/jiang24_split.csv" \
    "${jiang_dir}/jiang24_split.csv" \
    --size 41837463 \
    --sha256 5af7da86a5b3994d570c0b1957d91f17cebb9f9b1943bac738b74fdb14b2ef5d
}

case "${group}" in
  state) download_state ;;
  rpe1) download_rpe1 ;;
  jiang) download_jiang ;;
  jiang_h5ad) download_jiang_h5ad ;;
  all) download_state; download_rpe1; download_jiang; download_jiang_h5ad ;;
  *) echo "usage: $0 [state|rpe1|jiang|jiang_h5ad|all]" >&2; exit 2 ;;
esac
