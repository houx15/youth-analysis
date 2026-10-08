#!/bin/bash
#SBATCH --job-name=cultural-analysis
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=job.%j.cultural-analysis.out
# 北大集群资源参数互斥：只指定 mem，不同时指定任务数或 CPU 数。
# 内容行为及语义分析独立作业。
# 用法：sbatch run_cultural_analysis.sh build --input-dir cleaned_weibo_cov/2020 --vocabulary /path/to/vocab.json --output /path/to/new_run
set -euo pipefail
: "${SLURM_JOB_ID:?请通过 sbatch 提交此脚本}"
cd "${SLURM_SUBMIT_DIR:?}"
source "${CULTURAL_CONDA_INIT:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CULTURAL_CONDA_ENV:-opinion}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
python -m cultural_participation.research "$@"
