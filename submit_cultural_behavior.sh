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
# 不使用--parsable：Crane会忽略它并返回自然语言。
submit_job() {
    local response job_id
    response=$(sbatch "$@") || return $?
    printf '%s\n' "$response" >&2
    job_id=$(printf '%s\n' "$response" | sed -nE \
        -e 's/^[[:space:]]*Submitted batch job ([0-9]+)[[:space:]]*$/\1/p' \
        -e 's/^[[:space:]]*Job id allocated: ([0-9]+)\.?[[:space:]]*$/\1/p')
    if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
        echo "无法识别提交结果中的唯一作业ID。任务可能已经提交，请检查队列，勿重复提交。" >&2
        return 1
    fi
    printf '%s\n' "$job_id"
}

MONTH_JOB=""
REDUCE_JOB=""
if [[ $# -gt 0 ]]; then
    if [[ $# -ne 2 || ! "$2" =~ ^[0-9]+$ ]]; then
        echo "用法：bash $0 [--month-job 已提交的月份ID | --reduce-job 已提交的用户汇总ID]" >&2
        exit 1
    fi
    case "$1" in
        --month-job) MONTH_JOB="$2" ;;
        --reduce-job) REDUCE_JOB="$2" ;;
        *) echo "未知参数：$1" >&2; exit 1 ;;
    esac
fi
if [[ -z "$REDUCE_JOB" ]]; then
    if [[ -z "$MONTH_JOB" ]]; then
        MONTH_JOB=$(submit_job --array="1-12%$MONTH_CONCURRENCY" run_cultural_behavior.sh)
    fi
    echo "月度任务：$MONTH_JOB"
    REDUCE_JOB=$(submit_job --dependency="afterok:$MONTH_JOB" --array="0-$((BUCKET_COUNT - 1))%$REDUCE_CONCURRENCY" reduce_cultural_behavior.sh)
    echo "年度用户汇总：${REDUCE_JOB}，等待所有月份成功"
fi
SUMMARY_JOB=$(submit_job --dependency="afterok:$REDUCE_JOB" summarize_cultural_behavior.sh)
echo "统计汇总：${SUMMARY_JOB}，等待所有用户分片成功"
echo "结果：$OUTPUT_ROOT/$RUN_LABEL/summary"
