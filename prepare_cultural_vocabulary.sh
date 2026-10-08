#!/bin/bash
#SBATCH --job-name=cultural-vocab
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
set -euo pipefail
# 使用已安装 jieba 的 Python 环境，从仓库根目录提交。
: "${SLURM_JOB_ID:?请通过 sbatch 提交此脚本}"
cd "${SLURM_SUBMIT_DIR:?}"
input_dir="${1:?需要已解压的榜单目录}"
run_name="${2:?需要新的运行名称}"
if [[ ! "$run_name" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo '运行名称只能包含英文字母、数字、下划线和连字符' >&2
  exit 1
fi
output_dir="gender_norms/newspaper_data/cultural_participation/$run_name"
mkdir -p gender_norms/newspaper_data/cultural_participation
mkdir "$output_dir"
python -m cultural_participation collect --input-dir "$input_dir" --db "$output_dir/topics.sqlite"
python -m cultural_participation extract --db "$output_dir/topics.sqlite"
python -m cultural_participation export --db "$output_dir/topics.sqlite" --output "$output_dir/candidates.csv"
