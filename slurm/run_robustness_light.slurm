#!/bin/bash
#SBATCH --job-name=robust_light
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --mem=96G
#SBATCH --array=3-5
#SBATCH --output=job.%A_%a.robust_light.out
#SBATCH --mail-type=end
#SBATCH --mail-user=2401111059@stu.pku.edu.cn

# §13 稳健性套件的**三个轻族**：一个数组任务跑一个族。
#   3 samples        -> samples.parquet, sample_diagnostics.parquet,
#                       sample_balance.parquet          （§13.2 + §13.6 + §13.9）
#   4 measures       -> measures.parquet, measure_diagnostics.parquet
#                                                       （§13.1 + §13.8 + §13.7）
#   5 context_sample -> context_sample.parquet（供人工编码，§13.4）
#
# 另外两个族在 **slurm/run_robustness.slurm**（任务号 1/2）：
#   1 vocabulary（§13.3，默认 200 个 replicate，含重扫校准）
#   2 accounts（§13.5，约 326 次估计）
# 两个脚本**共用同一套任务编号**：`--array=4` 在这个脚本里是测量族，
# `--array=2` 在那个脚本里是账号族，任何一边都不会把编号错开。
#
# 用法（两个脚本都要投，缺一不可）：
#   export GENDER_DOMAIN_RUN_ID=$(date +%Y%m%d%H%M%S)   # 见下面"运行标识"
#   sbatch --export=ALL slurm/run_robustness.slurm
#   sbatch --export=ALL slurm/run_robustness_light.slurm
#   sbatch --export=ALL --array=4 slurm/run_robustness_light.slurm  # 只重跑测量族
#
# --------------------------------------------------------------------------
# 为什么单独有这个脚本
# --------------------------------------------------------------------------
# SLURM 的 --time / --mem 对**整个数组**是同一个值。这三个族原先和词表族、
# 账号族挤在同一个 1-5 的数组里，于是也按最贵的那个申请了 48 小时 / 192G：
# 五个任务同时排上队就要占住 960G，而其中三个一两个小时就结束。多占的额度
# 不会让它们算得更快，只会让整批作业更难排上队。
#
# 这里的 6 小时 / 96G / 4 核是这么定出来的：
#   * 任务 3（samples）与任务 4（measures）**只读表 C 与面板表**（measures
#     的 §13.7 留一月还是从 models_temporal 的输出里读，不重算），几个 GB
#     就够，是这三个里最便宜的两个；
#   * 任务 5（context_sample）是这个脚本的**内存上限所在**：它要在任务内部
#     建一份自己的 帖子×词 稀疏矩阵（复用 incidence.build_post_term_incidence，
#     与词表族同一套结构），96G 是按"这份矩阵 + 表 C"申请的。它比重脚本的
#     192G 少，是因为它不像账号族那样要在内存里反复持有改动过的表 C 副本
#     去跑几百次估计。
#   * 三个任务都在一两个小时内结束，6 小时留了三倍余量。
# 万一任务 5 真的被 OOM 杀掉（表 A 比预期大），**不要改这个脚本的 #SBATCH**，
# 单独把它按重族的额度重投一次即可（命令行上的 --mem 覆盖脚本里的 #SBATCH）：
#   sbatch --export=ALL --array=5 --mem=192G --time=12:00:00 slurm/run_robustness_light.slurm
#
# --------------------------------------------------------------------------
# 运行标识（run_id）：拆开之后必须**手工**导出一次
# --------------------------------------------------------------------------
# 各族的 manifest 里都记着 config.run_id()，它优先读环境变量
# GENDER_DOMAIN_RUN_ID，缺失时退化成 SLURM 的数组作业 id。五个族原先在同一个
# 数组作业里，%A 相同，于是五张结果表天然带着同一个标识；拆成两个数组作业
# 之后 %A 不再相同——**要让五张表还能被认出是同一批产出的，必须在投递之前
# 自己 export 一次 GENDER_DOMAIN_RUN_ID，并对两个脚本都 `--export=ALL`**。
#
# --------------------------------------------------------------------------
# 旋钮
# --------------------------------------------------------------------------
#   * 任务 3 的 bootstrap 次数：--n_replicates（SMP_REPLICATES，默认 200）
#   * 任务 5 每个分层格抽多少条：--n_per_cell（CTX_PER_CELL，默认 25）。
#     调低它会直接降低人工编码样本的精度（每格的 Wilson 区间会变宽），
#     不是一个"省机时"的旋钮，而是一个改变结论精度的旋钮。
# 例：
#   sbatch --export=ALL,SMP_REPLICATES=50 --array=3 slurm/run_robustness_light.slurm
#
# --------------------------------------------------------------------------
# 顺序要求
# --------------------------------------------------------------------------
# 任务 4（measures）的 §13.7 留一月重估从 models_temporal 的输出里读，不重算：
# 跑它之前 slurm/run_results.slurm 必须已经跑过，否则那一段会退化成一行注明
# 原因的行（LOMO_UNAVAILABLE_NOTE）。
#
# --------------------------------------------------------------------------
# 失败处理
# --------------------------------------------------------------------------
# set -euo pipefail：任何一步非零退出，这个数组任务立刻停下并把退出码传出去。
# 各族的结果表是逐变体/逐 replicate 增量落盘的（harness.append_rows 走临时
# 文件 + os.replace 原子改名），因此刻意**不**在开头清空
# analysis_data/robustness/。重跑某一族前如果想要一份干净的输出，请手动删掉
# 那一族自己的 parquet。

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opinion

YEAR="${YEAR:-2020}"
SMP_REPLICATES="${SMP_REPLICATES:-200}"
CTX_PER_CELL="${CTX_PER_CELL:-25}"

# 优先用调用者导出的运行标识（这是拆成两个脚本之后把五个族串成"同一批"的
# 唯一办法），没有才退回本数组作业的 id。
export GENDER_DOMAIN_RUN_ID="${GENDER_DOMAIN_RUN_ID:-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-$(date +%Y%m%d%H%M%S)}}}"
echo "GENDER_DOMAIN_RUN_ID=${GENDER_DOMAIN_RUN_ID} TASK=${SLURM_ARRAY_TASK_ID:-3}"

case "${SLURM_ARRAY_TASK_ID:-3}" in
  1|2)
    echo "任务 ${SLURM_ARRAY_TASK_ID} 属于重的两个族，请投 slurm/run_robustness.slurm" >&2
    exit 1
    ;;
  3)
    echo "=== 任务 3/5: §13.2 极端值 + §13.6 性别样本 + §13.9 用户类型 ==="
    python -m gender_domain.robustness.samples build \
      --year="${YEAR}" \
      --n_replicates="${SMP_REPLICATES}"
    ;;
  4)
    echo "=== 任务 4/5: §13.1 分母 + §13.8 帖子类型 + §13.7 时间限制 ==="
    # §13.7 的留一月重估从 models_temporal 的输出里读，不重算：跑这一任务
    # 之前 slurm/run_results.slurm 必须已经跑过，否则那一段会退化成一行
    # 注明原因的行（LOMO_UNAVAILABLE_NOTE）。
    python -m gender_domain.robustness.measures build --year="${YEAR}"
    ;;
  5)
    echo "=== 任务 5/5: §13.4 语境效度抽样（产出给人工编码的表）==="
    python -m gender_domain.robustness.context_sample build \
      --year="${YEAR}" \
      --n_per_cell="${CTX_PER_CELL}"
    ;;
  *)
    echo "未知的数组任务号 ${SLURM_ARRAY_TASK_ID:-}，本脚本只有 3-5" >&2
    exit 1
    ;;
esac

echo "任务 ${SLURM_ARRAY_TASK_ID:-3} 完成，run_id=${GENDER_DOMAIN_RUN_ID}"

# --------------------------------------------------------------------------
# 本脚本的三个任务与 slurm/run_robustness.slurm 的两个任务**全部**结束之后，
# 在登录节点跑综合层（它只读五张结果表，几秒钟）：
#   python3 -m gender_domain.robustness.synthesis build --year=2020
# 综合层刻意不放进任何一个数组：它必须等五个族都写完才有意义，而数组任务
# 之间没有顺序保证。真要串起来，用
#   --dependency=afterok:<重量数组作业 id>:<本数组作业 id>
# --------------------------------------------------------------------------
