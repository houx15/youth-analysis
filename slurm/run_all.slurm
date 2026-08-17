#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH -p C032M0128G
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32

# 全流程一次跑完：原始帖子 -> 四张分析表 -> 结果层 -> 绘图数据。
#
# 默认「已经算好的就跳过」。每个阶段先看自己的产物在不在，在就跳过并打印一行说明，
# 所以作业被墙钟杀掉之后直接重新 sbatch 即可，不会从头再算一遍。
#
# 用法：
#   sbatch slurm/run_all.slurm                 # 续跑：已有产物的阶段跳过
#   FORCE=1 sbatch slurm/run_all.slurm         # 全部重算（先删掉旧产物）
#   FORCE_STAGE=post_table sbatch slurm/run_all.slurm   # 只重算某一阶段及其下游
#   YEAR=2021 sbatch slurm/run_all.slurm       # 换年份
#
# FORCE_STAGE 可选值（按流水线顺序）：
#   post_table  retweet_table  user_tables  reconcile
#   profile_join  describe  models_core  interaction  combination  temporal  export
#
# run_id：结果层各步骤会往 analysis_data/results/run_stamps.json 里写同一个 run_id，
# 导出阶段据此拒绝把不同批次的结果表混在一起。本脚本把 run_id 存到
# analysis_data/.run_all_run_id，续跑时自动复用，所以不需要手工导出环境变量；
# FORCE=1 时重新生成一个。

set -euo pipefail

module load anaconda/3.11
source ~/.bash_profile
conda activate opinion

YEAR="${YEAR:-2020}"
FORCE="${FORCE:-0}"
FORCE_STAGE="${FORCE_STAGE:-}"

DATA_DIR="analysis_data"
RESULTS_DIR="${DATA_DIR}/results"
RUN_ID_FILE="${DATA_DIR}/.run_all_run_id"

mkdir -p "${DATA_DIR}"

# --- run_id：续跑复用，FORCE=1 重新生成 -------------------------------------
if [[ "${FORCE}" == "1" || ! -f "${RUN_ID_FILE}" ]]; then
    echo "${SLURM_JOB_ID:-$(date +%Y%m%d%H%M%S)}" > "${RUN_ID_FILE}"
fi
export GENDER_DOMAIN_RUN_ID="$(cat "${RUN_ID_FILE}")"

echo "============================================================"
echo "YEAR=${YEAR}  FORCE=${FORCE}  FORCE_STAGE=${FORCE_STAGE:-<none>}"
echo "GENDER_DOMAIN_RUN_ID=${GENDER_DOMAIN_RUN_ID}"
echo "开始时间: $(date '+%F %T')"
echo "============================================================"

# 一旦某个阶段被强制重算，它后面的阶段也必须重算，否则会把新旧产物混在一起
FORCING=0
if [[ "${FORCE}" == "1" ]]; then
    FORCING=1
fi

# should_run <阶段名> <产物路径...>
#   返回 0 表示要跑，1 表示跳过。产物任缺其一就跑。
should_run() {
    local stage="$1"; shift
    if [[ -n "${FORCE_STAGE}" && "${stage}" == "${FORCE_STAGE}" ]]; then
        FORCING=1
    fi
    if [[ "${FORCING}" == "1" ]]; then
        return 0
    fi
    local target
    for target in "$@"; do
        if [[ ! -e "${target}" ]]; then
            return 0
        fi
    done
    echo ""
    echo "[$(date '+%F %T')] 跳过阶段 ${stage}：产物已存在（$*）"
    return 1
}

stage_banner() {
    echo ""
    echo "------------------------------------------------------------"
    echo "[$(date '+%F %T')] 阶段开始: $*"
    echo "------------------------------------------------------------"
}

# 逐月分片：分片已存在就跳过该月
run_months() {
    local module="$1" shard_dir="$2" label="$3"
    local month mm shard
    for month in $(seq 1 12); do
        mm=$(printf "%02d" "${month}")
        shard="${DATA_DIR}/${shard_dir}_${YEAR}/month=${mm}.parquet"
        if [[ "${FORCING}" != "1" && -f "${shard}" ]]; then
            echo "[$(date '+%F %T')] 跳过 ${label} ${YEAR}-${mm}（分片已存在）"
            continue
        fi
        echo "[$(date '+%F %T')] ${label} ${YEAR}-${mm}"
        python -m "gender_domain.${module}" month --year="${YEAR}" --month="${month}"
    done
}

# ---------------------------------------------------------------
# 第一阶段：四张分析表
# ---------------------------------------------------------------
if should_run post_table "${DATA_DIR}/post_domain_measures_${YEAR}/month=12.parquet"; then
    stage_banner "表 A 帖子领域测量（逐月）"
    run_months build_post_table post_domain_measures "表 A"
fi

if should_run retweet_table "${DATA_DIR}/retweet_domain_events_${YEAR}/month=12.parquet"; then
    stage_banner "表 B 来源转发事件（逐月）"
    run_months build_retweet_table retweet_domain_events "表 B"
fi

if should_run user_tables "${DATA_DIR}/user_domain_${YEAR}.parquet" \
                          "${DATA_DIR}/user_month_domain_${YEAR}.parquet"; then
    stage_banner "表 C / 表 D 用户级母表与用户—月份面板"
    python -m gender_domain.build_user_tables build --year="${YEAR}"
fi

if should_run reconcile "${DATA_DIR}/reconciliation_${YEAR}.json"; then
    stage_banner "与既有 viz_data 基线对照"
    python -m gender_domain.reconcile_baseline run --year="${YEAR}"
fi

# ---------------------------------------------------------------
# 第二阶段：结果层
# ---------------------------------------------------------------
if should_run profile_join "${DATA_DIR}/user_domain_${YEAR}_with_profile.parquet"; then
    stage_banner "画像控制变量并入表 C"
    python -m gender_domain.profile_join build --year="${YEAR}"
fi

if should_run describe "${RESULTS_DIR}/table1_sample_activity.parquet" \
                       "${RESULTS_DIR}/table2_raw_gender_gaps.parquet"; then
    stage_banner "描述统计（表 1、表 2）"
    python -m gender_domain.describe build --year="${YEAR}"
fi

if should_run models_core "${RESULTS_DIR}/models_entry.parquet" \
                          "${RESULTS_DIR}/models_intensity.parquet" \
                          "${RESULTS_DIR}/models_share.parquet" \
                          "${RESULTS_DIR}/models_persistence.parquet"; then
    stage_banner "核心模型（进入、强度、比例、持续性）"
    python -m gender_domain.models_core build --year="${YEAR}"
fi

if should_run interaction "${RESULTS_DIR}/interaction_gender_domain.parquet"; then
    stage_banner "Gender x Domain 交互"
    python -m gender_domain.models_interaction build --year="${YEAR}"
fi

if should_run combination "${RESULTS_DIR}/decomposition_source_content.parquet" \
                          "${RESULTS_DIR}/combination_multinomial.parquet"; then
    stage_banner "来源—内容分解与参与组合"
    python -m gender_domain.models_combination build --year="${YEAR}"
fi

if should_run temporal "${RESULTS_DIR}/monthly_rates.parquet" \
                       "${RESULTS_DIR}/leave_one_month_out.parquet" \
                       "${RESULTS_DIR}/delay_quantiles.parquet"; then
    stage_banner "月度持续性与转发延迟"
    python -m gender_domain.models_temporal build --year="${YEAR}"
fi

# ---------------------------------------------------------------
# 第三阶段：可下载的绘图数据
# ---------------------------------------------------------------
if should_run export "${DATA_DIR}/figure_data/manifest.json"; then
    stage_banner "导出绘图数据"
    python -m gender_domain.export_figure_data build --year="${YEAR}"
fi

echo ""
echo "============================================================"
echo "全部阶段完成"
echo "结束时间: $(date '+%F %T')"
echo "GENDER_DOMAIN_RUN_ID=${GENDER_DOMAIN_RUN_ID}"
echo ""
echo "下载以下目录到本地后绘图："
echo "  ${DATA_DIR}/figure_data/"
echo "  ${RESULTS_DIR}/"
echo "然后本地执行： python3 -m gender_domain.figures all --year=${YEAR}"
echo "============================================================"
