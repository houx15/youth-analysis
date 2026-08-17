#!/bin/bash
#SBATCH --job-name=genderall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=job.%j.genderall.out
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=2401111059@stu.pku.edu.cn

# 全流程一次跑完：原始帖子 -> 四张分析表 -> 结果层 -> 绘图数据 -> §13 稳健性套件。
#
# 默认「已经算好的就跳过」。每个阶段先看自己的产物在不在，在就跳过并打印一行说明，
# 所以作业被墙钟杀掉之后直接重新 sbatch 即可，不会从头再算一遍。
#
# 最后一个阶段与前面几个不同：它**提交**稳健性的两个数组作业与一个综合层作业
# （见下面第四阶段的说明），而不是在本作业里顺序跑完。因此本作业结束时稳健性
# 通常还在排队或运行——`squeue -u $USER` 看进度，别把本作业的"完成"读成
# 稳健性也完成了。
#
# 用法：
#   sbatch slurm/run_all.slurm                 # 续跑：已有产物的阶段跳过
#   FORCE=1 sbatch slurm/run_all.slurm         # 全部重算（先删掉旧产物）
#   FORCE_STAGE=post_table sbatch slurm/run_all.slurm   # 只重算某一阶段及其下游
#   YEAR=2021 sbatch slurm/run_all.slurm       # 换年份
#   ROBUSTNESS=0 sbatch slurm/run_all.slurm    # 只要主结果，不投稳健性
#
# FORCE_STAGE 可选值（按流水线顺序）：
#   post_table  retweet_table  user_tables  reconcile
#   profile_join  describe  models_core  interaction  combination  temporal  export
#   robustness
#
# 内存：整条流水线的峰值出现在结果层（约 192G），逐月分片阶段只要几十 G。
# 本集群不接受 --mem，内存跟着分区走，所以这里记的是实测占用而非申请额度——
# 要挑一个装得下 192G 的分区投，否则跑到结果层才被 OOM 杀掉。
#
# run_id：结果层各步骤会往 analysis_data/results/run_stamps.json 里写同一个 run_id，
# 导出阶段据此拒绝把不同批次的结果表混在一起。本脚本把 run_id 存到
# analysis_data/.run_all_run_id，续跑时自动复用，所以不需要手工导出环境变量；
# FORCE=1 时重新生成一个。

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
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

# ---------------------------------------------------------------
# 第四阶段：§13 稳健性套件（投出去，不在本作业里跑）
# ---------------------------------------------------------------
# 这一阶段**提交**三个作业，而不是顺序执行五个族。理由是额度：
# 词表族与账号族各自可能跑满十几个小时，五个族串起来要三十小时上下，叠在
# 前面三个阶段之后必然撞穿 48 小时的墙钟；而它们本来就是互相独立的，
# 拆成两个数组并行跑，墙钟只取决于最慢的那一族。把它们塞进本作业顺序执行，
# 等于把 slurm/run_robustness*.slurm 拆成两个脚本的理由整个丢掉。
#
# 依赖链：重量数组 + 轻量数组 -> 综合层（afterok，两个都成功才跑）。
# 五个族共用本作业的 run_id，所以事后能证明它们与主结果属于同一批。
#
# ROBUSTNESS=0 可以跳过这一阶段（只想要主结果时用）。
ROBUSTNESS="${ROBUSTNESS:-1}"
ROBUST_DIR="${DATA_DIR}/robustness"

if [[ "${ROBUSTNESS}" != "1" ]]; then
    echo ""
    echo "[$(date '+%F %T')] 跳过稳健性套件（ROBUSTNESS=${ROBUSTNESS}）"
elif should_run robustness "${ROBUST_DIR}/synthesis.parquet"; then
    stage_banner "§13 稳健性套件（提交三个作业）"
    # sbatch 从计算节点提交在多数集群上是允许的；万一本集群不允许，不能让它
    # 把一个已经跑完全部主结果的作业判成失败——整段兜住，打印手工投递的命令，
    # 退出码仍然是 0。三次提交任意一次失败都算整段失败：只投出去一半更糟，
    # 综合层的 afterok 会永远等不到另一半，作业挂在队列里没人发现。
    submit_robustness() {
        local heavy light syn
        heavy=$(sbatch --parsable --export=ALL slurm/run_robustness.slurm) \
            || return 1
        light=$(sbatch --parsable --export=ALL slurm/run_robustness_light.slurm) \
            || return 1
        syn=$(sbatch --parsable --export=ALL \
                  --dependency="afterok:${heavy}:${light}" \
                  slurm/run_robustness_synthesis.slurm) || return 1
        echo "已提交：重量数组 ${heavy}、轻量数组 ${light}、综合层 ${syn}"
        echo "综合层会等前两个数组都成功之后才启动。"
    }

    if ! submit_robustness; then
        echo "从计算节点提交稳健性作业失败，主结果不受影响。" >&2
        echo "请在登录节点手工投递（run_id 要沿用本次的，否则事后对不上批次）：" >&2
        echo "  export GENDER_DOMAIN_RUN_ID=${GENDER_DOMAIN_RUN_ID}" >&2
        echo "  h=\$(sbatch --parsable --export=ALL slurm/run_robustness.slurm)" >&2
        echo "  l=\$(sbatch --parsable --export=ALL slurm/run_robustness_light.slurm)" >&2
        echo "  sbatch --export=ALL --dependency=afterok:\$h:\$l slurm/run_robustness_synthesis.slurm" >&2
    fi
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
echo ""
echo "稳健性套件是另投的作业，本作业结束时它多半还在跑："
echo "  squeue -u \$USER            # 看 robust_heavy / robust_light / robustsyn"
echo "  ${ROBUST_DIR}/  # 五个族 + synthesis*.parquet 都落在这里"
echo "============================================================"
