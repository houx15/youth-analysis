#!/bin/bash
#SBATCH -o job.%j.cultural_analysis.out
#SBATCH -p C032M0128G
#SBATCH --qos=low
#SBATCH -J cultural_analysis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
# 调度参数沿用用户已验证可运行的北大集群模板。
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
