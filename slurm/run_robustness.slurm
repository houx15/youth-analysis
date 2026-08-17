#!/bin/bash
#SBATCH --job-name=robust
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=192G
#SBATCH --array=1-5
#SBATCH --output=job.%A_%a.robust.out
#SBATCH --mail-type=end
#SBATCH --mail-user=2401111059@stu.pku.edu.cn

# §13 稳健性套件：一个数组任务跑一个族。
# 输出（全部只写 analysis_data/robustness/）：
#   1 vocabulary     -> vocabulary.parquet, vocabulary_diagnostics.parquet
#   2 accounts       -> accounts.parquet, account_diagnostics.parquet,
#                       account_concentration.parquet
#   3 samples        -> samples.parquet, sample_diagnostics.parquet,
#                       sample_balance.parquet
#   4 measures       -> measures.parquet, measure_diagnostics.parquet
#   5 context_sample -> context_sample.parquet（供人工编码）
# 各族自己写各自的 manifest 子目录。综合层（synthesis）**不在这个数组里**，
# 见文件末尾的运行顺序。
#
# 用法：
#   sbatch slurm/run_robustness.slurm
#   sbatch --array=2 slurm/run_robustness.slurm          # 只重跑账号族
#   sbatch --array=1 slurm/run_robustness.slurm          # 只重跑词表族
#
# --------------------------------------------------------------------------
# 稀疏关联结构在**每个任务内部**各建一份
# --------------------------------------------------------------------------
# incidence.build_post_term_incidence / build_user_account_incidence 是词表族
# 与账号族全部重复抽样的前提（没有它，每个 replicate 要重扫一年正文或重聚合
# 三千多万行）。它由各族自己的 build_context() 在**任务内部**建一次、
# 跨全部 replicate 复用——刻意不做成"先跑一个任务建好、其它任务读盘"：
# 数组任务之间没有顺序保证，一个共享的中间产物只会带来"读到半成品"这一类
# 无法复现的失败，而重建一次的代价（几分钟 + 内存）远低于这个风险。
# 因此 --mem 是按"单个任务内存里同时装得下一份矩阵加一份表 C"申请的。
#
# --------------------------------------------------------------------------
# 额度是按最贵的那两个族申请的
# --------------------------------------------------------------------------
# 任务 1（词表）：默认 200 个 replicate，每个 replicate 一次
#   harness.estimate_all（六个量 × 两层，含两次 DiD 拟合），外加默认 20 个
#   replicate 的重扫校准——校准要回 cleaned_weibo_cov 读日文件、重扫几万条
#   正文，是整个套件里唯一还碰原始文本的一步。
# 任务 2（账号）：默认 326 次 estimate_all（留一账号 2×50、留一类别 8+11、
#   剔头部 2×3、中央级媒体 1、账号 bootstrap 200）。
# 这两个任务各自都可能跑满十几个小时；任务 3/4/5 通常一两个小时内结束，
# 但 SLURM 的 --time/--mem 对整个数组是同一个值，只能按最贵的那个申请。
#
# 想让它跑得更快、更省：**调低下面两个旋钮，不要改代码**。
#   * 词表族的 replicate 数：--n_replicates=50（默认 200，方案文档允许
#     200–500；50 只够看方向，不足以报分布）。校准子样本另有
#     --n_calibration（默认 20）、随机检验 --n_probe（默认 500）。
#   * 账号族的留一账号个数：--top_n=20（默认 50）。**调低它会同时降低这次
#     留一检验覆盖的转发量份额**，而那个份额会被写进诊断表与 manifest——
#     调低之后请照着新的 coverage 读结果，不要照着 50 个账号的说法读。
#     账号 bootstrap 的次数是 --n_replicates（默认 200）。
# 例：
#   sbatch --array=1 --export=ALL,VOC_REPLICATES=50 slurm/run_robustness.slurm
#   sbatch --array=2 --export=ALL,ACC_TOP_N=20,ACC_REPLICATES=50 slurm/run_robustness.slurm
#
# --------------------------------------------------------------------------
# 失败处理
# --------------------------------------------------------------------------
# set -euo pipefail：任何一步非零退出，这个数组任务立刻停下并把退出码传出去。
# 各族的结果表是**逐变体/逐 replicate 增量落盘**的（harness.append_rows 走
# 临时文件 + os.replace 原子改名），撞墙钟被杀时盘上留下的是"已经跑完的那些
# replicate"，因此刻意**不**在开头清空 analysis_data/robustness/。
# 重跑某一族前如果想要一份干净的输出，请手动删掉那一族自己的 parquet。

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opinion

YEAR="${YEAR:-2020}"
VOC_REPLICATES="${VOC_REPLICATES:-200}"
VOC_CALIBRATION="${VOC_CALIBRATION:-20}"
ACC_TOP_N="${ACC_TOP_N:-50}"
ACC_REPLICATES="${ACC_REPLICATES:-200}"
SMP_REPLICATES="${SMP_REPLICATES:-200}"
CTX_PER_CELL="${CTX_PER_CELL:-25}"

# 同一次数组作业的五个任务共用一个运行标识（%A 是数组作业 id，对全部任务
# 相同），这样五张结果表能被认出是同一批产出的。
export GENDER_DOMAIN_RUN_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-$(date +%Y%m%d%H%M%S)}}"
echo "GENDER_DOMAIN_RUN_ID=${GENDER_DOMAIN_RUN_ID} TASK=${SLURM_ARRAY_TASK_ID:-1}"

case "${SLURM_ARRAY_TASK_ID:-1}" in
  1)
    echo "=== 任务 1/5: §13.3 词表稳健性（最贵，含重扫校准）==="
    python -m gender_domain.robustness.vocabulary build \
      --year="${YEAR}" \
      --n_replicates="${VOC_REPLICATES}" \
      --n_calibration="${VOC_CALIBRATION}"
    ;;
  2)
    echo "=== 任务 2/5: §13.5 来源账号稳健性（约 326 次估计）==="
    python -m gender_domain.robustness.accounts build \
      --year="${YEAR}" \
      --top_n="${ACC_TOP_N}" \
      --n_replicates="${ACC_REPLICATES}"
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
    echo "未知的数组任务号 ${SLURM_ARRAY_TASK_ID:-}，本数组只有 1-5" >&2
    exit 1
    ;;
esac

echo "任务 ${SLURM_ARRAY_TASK_ID:-1} 完成，run_id=${GENDER_DOMAIN_RUN_ID}"

# --------------------------------------------------------------------------
# 五个任务全部结束之后，在登录节点跑综合层（它只读五张结果表，几秒钟）：
#   python3 -m gender_domain.robustness.synthesis build --year=2020
# 综合层刻意不放进这个数组：它必须等五个族都写完才有意义，而数组任务之间
# 没有顺序保证。真要串起来，用 --dependency=afterok:<数组作业 id>。
# --------------------------------------------------------------------------
