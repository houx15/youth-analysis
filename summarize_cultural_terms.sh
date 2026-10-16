#!/bin/bash
#SBATCH -o job.%A.%a.term_summary.out
#SBATCH -p C032M0128G
#SBATCH --qos=low
#SBATCH -J term_summary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8

module load anaconda/3.11
source ~/.bash_profile
conda activate opinion || exit 1
set -e
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source cultural_participation/behavior_job.conf
RUN_DIR="$OUTPUT_ROOT/$RUN_LABEL"
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
python -u -m cultural_participation.term_behavior summarize \
    --input-dir "$RUN_DIR/term_behavior_v1" --output "$RUN_DIR/term_summary_v1"
tar -czf "$RUN_DIR/term_summary_review.tar.gz" -C "$RUN_DIR/term_summary_v1" \
    term_descriptives.csv term_gender_gaps.csv term_coverage.csv manifest.json
echo "请下载：$PWD/$RUN_DIR/term_summary_review.tar.gz"
