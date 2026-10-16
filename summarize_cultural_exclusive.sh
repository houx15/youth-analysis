#!/bin/bash
#SBATCH -o job.%j.cultural_exclusive_summary.out
#SBATCH -p C032M0128G
#SBATCH --qos=low
#SBATCH -J exclusive_summary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
module load anaconda/3.11
source ~/.bash_profile
conda activate opinion || exit 1
set -e
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source cultural_participation/extended_job.conf
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
ROOT="$OUTPUT_ROOT/$RUN_LABEL"
python -u -m cultural_participation.extended_behavior prepare \
  --full-run "$ROOT/exclusive_full_v1" --top-run "$ROOT/exclusive_top250_v1" \
  --output "$ROOT/sample" --profile-csv "$PROFILE_CSV" --id-mapping "$ID_MAPPING" \
  --year "$YEAR" --match-width "$MATCH_WIDTH"
for version in exclusive_full_v1 exclusive_top250_v1; do
    python -u -m cultural_participation.extended_behavior summarize \
      --input-dir "$ROOT/$version" --sample-dir "$ROOT/sample" --output "$ROOT/$version/extended_summary"
    python -u -m cultural_participation.extended_behavior annual \
      --input-dir "$ROOT/$version" --output "$ROOT/$version/annual_summary"
done
tar --exclude='*/user_domain.csv' -czf "$ROOT/exclusive_review.tar.gz" -C "$ROOT" \
  sample/manifest.json sample/matching_balance.csv sample/sample_coverage.csv \
  sample/age_distribution.csv sample/profile_coverage.csv sample/stratified_balance.csv \
  exclusive_full_v1/extended_summary exclusive_full_v1/annual_summary \
  exclusive_top250_v1/extended_summary exclusive_top250_v1/annual_summary
echo "完成，请下载：$PWD/$ROOT/exclusive_review.tar.gz"
