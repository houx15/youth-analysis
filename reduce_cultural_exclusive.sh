#!/bin/bash
#SBATCH -o job.%A.%a.cultural_exclusive_reduce.out
#SBATCH -p C032M0128G
#SBATCH --qos=low
#SBATCH -J exclusive_reduce
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --array=0-63%4
module load anaconda/3.11
source ~/.bash_profile
conda activate opinion || exit 1
set -e
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source cultural_participation/extended_job.conf
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for version in exclusive_full_v1 exclusive_top250_v1; do
    python -u -m cultural_participation.extended_behavior reduce \
      --input-dir "$OUTPUT_ROOT/$RUN_LABEL/$version" \
      --bucket "${SLURM_ARRAY_TASK_ID:?需要数组任务}"
done
