#!/bin/bash
#SBATCH -o job.%j.cultural_behavior.out
#SBATCH -p C032M0128G
#SBATCH --qos=low
#SBATCH -J cultural_behavior
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8

module load anaconda/3.11
source ~/.bash_profile
conda activate opinion || exit 1

set -e
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source cultural_participation/behavior_job.conf
[ "$MAX_ROWS" -ge 0 ]
python -c "import numpy, pyarrow"
RUN_DIR="$OUTPUT_ROOT/${RUN_LABEL}_${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
LIMIT_ARGS=()
if [ "$MAX_ROWS" -gt 0 ]; then
    LIMIT_ARGS=(--max-rows "$MAX_ROWS")
fi
echo "输出目录：$RUN_DIR"
python -u -m cultural_participation.research build \
    --input-dir "$INPUT_DIR" --pattern "$PATTERN" --year "$YEAR" \
    --vocabulary "$VOCABULARY" --output "$RUN_DIR/cache" \
    --batch-size "$BATCH_SIZE" "${LIMIT_ARGS[@]}"
echo "命中缓存完成，开始用户汇总与性别比较。"
python -u -m cultural_participation.research summarize \
    --db "$RUN_DIR/cache/behavior.sqlite" --output "$RUN_DIR/summary"
echo "全部完成：$RUN_DIR/summary"
