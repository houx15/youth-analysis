#!/bin/bash
# 这里只提交依赖作业，不在登录节点执行数据处理。
set -euo pipefail
cd "$(dirname "$0")"
source cultural_participation/behavior_job.conf
[[ "$BUCKET_COUNT" =~ ^[1-9][0-9]*$ ]]
[[ "$MONTH_CONCURRENCY" =~ ^[1-9][0-9]*$ ]]
[[ "$REDUCE_CONCURRENCY" =~ ^[1-9][0-9]*$ ]]
[[ -d "$INPUT_DIR" && -f "$VOCABULARY" ]]
if [[ -e "$OUTPUT_ROOT/$RUN_LABEL/summary" ]]; then
    echo "汇总目录已存在；请确认已有结果，或修改RUN_LABEL。" >&2
    exit 1
fi
MONTH_JOB=$(sbatch --parsable --array="1-12%$MONTH_CONCURRENCY" run_cultural_behavior.sh)
MONTH_JOB=${MONTH_JOB%%;*}
echo "月度任务：$MONTH_JOB"
REDUCE_JOB=$(sbatch --parsable --dependency="afterok:$MONTH_JOB" --array="0-$((BUCKET_COUNT - 1))%$REDUCE_CONCURRENCY" reduce_cultural_behavior.sh)
REDUCE_JOB=${REDUCE_JOB%%;*}
echo "年度用户汇总：${REDUCE_JOB}，等待所有月份成功"
SUMMARY_JOB=$(sbatch --parsable --dependency="afterok:$REDUCE_JOB" summarize_cultural_behavior.sh)
echo "统计汇总：${SUMMARY_JOB}，等待所有用户分片成功"
echo "结果：$OUTPUT_ROOT/$RUN_LABEL/summary"
