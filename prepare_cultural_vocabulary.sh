#!/bin/bash
#SBATCH -o job.%j.cultural_vocab.out
#SBATCH -p C032M0128G
#SBATCH --qos=low
#SBATCH -J cultural_vocab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
# 调度参数沿用用户已验证可运行的北大集群模板。
# 用法：在仓库根目录直接 sbatch prepare_cultural_vocabulary.sh
# 路径和规模配置：cultural_participation/vocabulary_job.conf
set -euo pipefail
: "${SLURM_JOB_ID:?请通过 sbatch 提交此脚本}"
cd "${SLURM_SUBMIT_DIR:?}"
source cultural_participation/vocabulary_job.conf
source "${CULTURAL_CONDA_INIT:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CULTURAL_CONDA_ENV:-opinion}"
if [[ ! "$MAX_LINES" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo 'MAX_LINES 必须是非负整数，0 表示全量' >&2
  exit 1
fi
if [[ ! "$RUN_LABEL" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo 'RUN_LABEL 只能包含英文字母、数字、下划线和连字符' >&2
  exit 1
fi
run_name="${RUN_LABEL}_${SLURM_JOB_ID}"
output_dir="gender_norms/newspaper_data/cultural_participation/$run_name"
printf '输入目录：%s\n输出目录：%s\n最多读取行数：%s（0表示全量）\n' "$INPUT_DIR" "$output_dir" "$MAX_LINES"
python -c 'import jieba; print("jieba", jieba.__version__)'
mkdir -p gender_norms/newspaper_data/cultural_participation
mkdir "$output_dir"
# 用非空命令数组，兼容较旧 Bash 的 nounset 行为。
collect_command=(python -m cultural_participation collect --input-dir "$INPUT_DIR" --db "$output_dir/topics.sqlite")
if [[ "$MAX_LINES" != 0 ]]; then
  collect_command+=(--max-lines "$MAX_LINES")
fi
"${collect_command[@]}"
python -m cultural_participation extract --db "$output_dir/topics.sqlite"
python -m cultural_participation export --db "$output_dir/topics.sqlite" --output "$output_dir/candidates.csv"
