#!/bin/bash
#SBATCH -o job.%A.%a.cultural_month.out
#SBATCH -p C032M0128G
#SBATCH --qos=low
#SBATCH -J cultural_month
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --array=1-12%3
module load anaconda/3.11
source ~/.bash_profile
conda activate opinion || exit 1
set -e
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source cultural_participation/behavior_job.conf
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN_DIR="$OUTPUT_ROOT/$RUN_LABEL"
python -u -m cultural_participation.parquet_behavior month \
    --input-dir "$INPUT_DIR" --vocabulary "$VOCABULARY" --year "$YEAR" \
    --month "${SLURM_ARRAY_TASK_ID:?请使用submit_cultural_behavior.sh提交}" \
    --bucket-count "$BUCKET_COUNT" --batch-size "$BATCH_SIZE" --output "$RUN_DIR/months"
