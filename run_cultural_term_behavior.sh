#!/bin/bash
#SBATCH -o job.%A.%a.cultural_terms.out
#SBATCH -p C032M0128G
#SBATCH --qos=low
#SBATCH -J cultural_terms
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --array=0-63%4
module load anaconda/3.11
source ~/.bash_profile
conda activate opinion || exit 1
set -e
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source cultural_participation/behavior_job.conf
RUN_DIR="$OUTPUT_ROOT/$RUN_LABEL"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u -m cultural_participation.term_behavior build \
    --input-dir "$RUN_DIR" --output "$RUN_DIR/term_behavior_v1" \
    --bucket "${SLURM_ARRAY_TASK_ID:?需要数组分片编号}" --vocabulary "$VOCABULARY"
