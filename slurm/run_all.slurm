#!/bin/bash
#SBATCH --job-name=genderall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=192G
#SBATCH --output=job.%j.genderall.out
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=2401111059@stu.pku.edu.cn

# The whole pipeline in one job: raw posts -> four analysis tables -> results -> figure data.
#
# This replaces submitting six scripts and hand-managing their order. The per-stage scripts
# are kept alongside it for targeted re-runs (re-running one failed month is much cheaper
# with `sbatch --array=7 slurm/build_post_table.slurm` than restarting everything).
#
# The month shards run SEQUENTIALLY here rather than as a 12-way array, so this job trades
# wall clock for simplicity. The resources below are the maximum any single stage needs, held
# for the whole run.
#
# Usage:
#   sbatch slurm/run_all.slurm
#   YEAR=2021 sbatch slurm/run_all.slurm                       # a different year
#   RESUME=1 GENDER_DOMAIN_RUN_ID=<id> sbatch slurm/run_all.slurm   # continue a killed run
#
# Resuming: pass RESUME=1 to skip month shards that already exist, and pass the ORIGINAL
# run id so the results written by both attempts carry one stamp. Without that id the export
# stage will correctly refuse to mix two runs' result tables. The id of any run is printed
# below and appears in every manifest.

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opinion

YEAR="${YEAR:-2020}"
RESUME="${RESUME:-0}"
export GENDER_DOMAIN_RUN_ID="${GENDER_DOMAIN_RUN_ID:-${SLURM_JOB_ID:-$(date +%Y%m%d%H%M%S)}}"

echo "============================================================"
echo "YEAR=${YEAR}  RESUME=${RESUME}"
echo "GENDER_DOMAIN_RUN_ID=${GENDER_DOMAIN_RUN_ID}"
echo "开始时间: $(date '+%F %T')"
echo "============================================================"

stage() {
    echo ""
    echo "------------------------------------------------------------"
    echo "[$(date '+%F %T')] 阶段开始: $*"
    echo "------------------------------------------------------------"
}

# 逐月处理：分片已存在且 RESUME=1 时跳过，便于长作业被墙钟杀掉后继续
run_months() {
    local module="$1" shard_dir="$2" label="$3"
    for month in $(seq 1 12); do
        local mm shard
        mm=$(printf "%02d" "${month}")
        shard="analysis_data/${shard_dir}_${YEAR}/month=${mm}.parquet"
        if [[ "${RESUME}" == "1" && -f "${shard}" ]]; then
            echo "[$(date '+%F %T')] 跳过 ${label} ${YEAR}-${mm}（分片已存在，RESUME=1）"
            continue
        fi
        echo "[$(date '+%F %T')] ${label} ${YEAR}-${mm}"
        python -m "gender_domain.${module}" month --year="${YEAR}" --month="${month}"
    done
}

# ---------------------------------------------------------------
# 第一阶段：四张分析表
# ---------------------------------------------------------------
stage "表 A 帖子领域测量（逐月）"
run_months build_post_table post_domain_measures "表 A"

stage "表 B 来源转发事件（逐月）"
run_months build_retweet_table retweet_domain_events "表 B"

stage "表 C / 表 D 用户级母表与用户—月份面板"
python -m gender_domain.build_user_tables build --year="${YEAR}"

stage "与既有 viz_data 基线对照"
python -m gender_domain.reconcile_baseline run --year="${YEAR}"

# ---------------------------------------------------------------
# 第二阶段：结果层
# ---------------------------------------------------------------
stage "画像控制变量并入表 C"
python -m gender_domain.profile_join build --year="${YEAR}"

stage "描述统计（表 1、表 2）"
python -m gender_domain.describe build --year="${YEAR}"

stage "核心模型（进入、强度、比例、持续性）"
python -m gender_domain.models_core build --year="${YEAR}"

stage "Gender x Domain 交互"
python -m gender_domain.models_interaction build --year="${YEAR}"

stage "来源—内容分解与参与组合"
python -m gender_domain.models_combination build --year="${YEAR}"

stage "月度持续性与转发延迟"
python -m gender_domain.models_temporal build --year="${YEAR}"

# ---------------------------------------------------------------
# 第三阶段：可下载的绘图数据
# ---------------------------------------------------------------
stage "导出绘图数据"
python -m gender_domain.export_figure_data build --year="${YEAR}"

echo ""
echo "============================================================"
echo "全部阶段完成"
echo "结束时间: $(date '+%F %T')"
echo "GENDER_DOMAIN_RUN_ID=${GENDER_DOMAIN_RUN_ID}"
echo ""
echo "下载以下目录到本地后绘图："
echo "  analysis_data/figure_data/"
echo "  analysis_data/results/"
echo "然后本地执行： python3 -m gender_domain.figures all --year=${YEAR}"
echo "============================================================"
