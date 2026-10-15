#!/bin/bash
#SBATCH -o job.%A.%a.cultural_summary.out
#SBATCH -p C032M0128G
#SBATCH --qos=low
#SBATCH -J cultural_summary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8

module load anaconda/3.11
source ~/.bash_profile
conda activate opinion || exit 1
set -e
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source cultural_participation/behavior_job.conf
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN_DIR="$OUTPUT_ROOT/$RUN_LABEL"
python -u -m cultural_participation.parquet_behavior summarize \
    --input-dir "$RUN_DIR/users" --output "$RUN_DIR/summary"
