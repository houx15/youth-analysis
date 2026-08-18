#!/bin/bash
#SBATCH --job-name=robust_heavy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=48:00:00
#SBATCH --array=1-2
#SBATCH --output=job.%A_%a.robust.out
#SBATCH --mail-type=end
#SBATCH --mail-user=2401111059@stu.pku.edu.cn

# §13 稳健性套件的**两个贵族**：一个数组任务跑一个族。
#   1 vocabulary     -> vocabulary.parquet, vocabulary_diagnostics.parquet
#   2 accounts       -> accounts.parquet, account_diagnostics.parquet,
#                       account_concentration.parquet
#
# 另外三个族在 **slurm/run_robustness_light.slurm**（任务号 3/4/5，见下）：
#   3 samples        -> samples.parquet, sample_diagnostics.parquet,
#                       sample_balance.parquet
#   4 measures       -> measures.parquet, measure_diagnostics.parquet
#   5 context_sample -> context_sample.parquet（供人工编码）
# 两个脚本**共用同一套任务编号**：`--array=2` 在这个脚本里是账号族，
# `--array=4` 在那个脚本里是测量族，任何一边都不会把编号错开。
# 各族自己写各自的 manifest 子目录。综合层（synthesis）**两个数组都不含**，
# 见文件末尾的运行顺序。
#
# 用法（两个脚本都要投，缺一不可）：
#   export GENDER_DOMAIN_RUN_ID=$(date +%Y%m%d%H%M%S)   # 见下面"运行标识"
#   sbatch --export=ALL slurm/run_robustness.slurm
#   sbatch --export=ALL slurm/run_robustness_light.slurm
#   sbatch --export=ALL --array=2 slurm/run_robustness.slurm   # 只重跑账号族
#   sbatch --export=ALL --array=1 slurm/run_robustness.slurm   # 只重跑词表族
#
# --------------------------------------------------------------------------
# 为什么要拆成两个脚本
# --------------------------------------------------------------------------
# SLURM 的 --time / --cpus-per-task 对**整个数组**是同一个值。五个族原先挤
# 在一个 1-5 的数组里，就只能按最贵的那个申请：48 小时 / 8 核占满五个任务，
# 而其中三个族一两个小时就结束、峰值内存也远用不到重族的量级。多占的那部分
# 不会变成算得更快，只会让这批作业更难排上队，也挡着别人的作业。
# 因此重的两族留在这里（48 小时），轻的三族去 slurm/run_robustness_light.slurm
# （24 小时）。两个脚本的核数现在都按实测内存取 12 核，见下面那张表。
#
# --------------------------------------------------------------------------
# 额度是**实测**定出来的，不是估的（2026-08-18 那批作业的 sacct）
# --------------------------------------------------------------------------
# 关键机制：本集群不接受 --mem，**内存 = cpus-per-task x 4000M**（默认分区
# C064M0256G：256000M / 64 核）。所以核数在这里同时是内存旋钮——申请核不是
# 为了并行（statsmodels 是单线程的），是为了买内存。
#
#   族              MaxRSS    墙钟        结果        瓶颈
#   accounts        2.28 GB   22:15:46    COMPLETED   时间
#   samples         1.88 GB   06:00:12    TIMEOUT     时间
#   vocabulary      >32 GB    00:04:04    OOM         内存
#   measures        >16 GB    00:01:54    OOM         内存
#   context_sample  >16 GB    00:01:54    OOM         内存
#
# 三个 OOM 的族是且只是调用 incidence.build_post_term_incidence 的三个。那个
# 函数原来把逐列压缩写在 pd.concat **后面**，循环里堆的是 object 字符串，
# 1.47 亿帖下累计约 29.5 GB、concat 峰值约 59 GB，谁都没活到压缩那一行。
# 压缩挪进循环之后峰值降到约 17 GB（measures 不载入 weibo_id，约 10 GB）。
#
# 于是核数按"装得下 17 GB 再留一倍余量"取 12 核 = 48000M。此前的注释写着
# 192G / 96G，那两个数字是估的，与实测差了两个数量级——accounts 真正只用了
# 2.28 GB。别再照着它们挑分区。
#
# 真要更大内存：C064M1024G 分区是 1030000M / 64 核 = 16093M 每核，同样核数
# 拿到四倍内存（-p C064M1024G）。目前用不着。
#
# --------------------------------------------------------------------------
# 运行标识（run_id）：拆开之后必须**手工**导出一次
# --------------------------------------------------------------------------
# 各族的 manifest 里都记着 config.run_id()，它优先读环境变量
# GENDER_DOMAIN_RUN_ID，缺失时退化成 SLURM 的数组作业 id。五个族原先在同一个
# 数组作业里，%A 相同，于是五张结果表天然带着同一个标识；拆成两个数组作业
# 之后 %A 不再相同——**要让五张表还能被认出是同一批产出的，必须在投递之前
# 自己 export 一次 GENDER_DOMAIN_RUN_ID，并对两个脚本都 `--export=ALL`**。
# 不这么做不会报错，只会让两半的 manifest 各带各的标识，事后没有人能证明
# 它们属于同一批。
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
# 这份结构的实测峰值见下表：压缩修好之后约 17 GB，12 核（48000M）装得下。
#
# --------------------------------------------------------------------------
# 额度是按这两个族申请的
# --------------------------------------------------------------------------
# 任务 1（词表）：默认 200 个 replicate，每个 replicate 一次
#   harness.estimate_all（六个量 × 两层，含两次 DiD 拟合），外加默认 20 个
#   replicate 的重扫校准——校准要回 cleaned_weibo_cov 读日文件、重扫几万条
#   正文，是整个套件里唯一还碰原始文本的一步。
# 任务 2（账号）：默认 326 次 estimate_all（留一账号 2×50、留一类别 8+11、
#   剔头部 2×3、中央级媒体 1、账号 bootstrap 200）。
# 这两个任务各自都可能跑满十几个小时。
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

# 优先用调用者导出的运行标识（这是拆成两个脚本之后把五个族串成"同一批"的
# 唯一办法），没有才退回本数组作业的 id。
export GENDER_DOMAIN_RUN_ID="${GENDER_DOMAIN_RUN_ID:-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-$(date +%Y%m%d%H%M%S)}}}"
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
  3|4|5)
    echo "任务 ${SLURM_ARRAY_TASK_ID} 属于轻量的三个族，请投 slurm/run_robustness_light.slurm" >&2
    exit 1
    ;;
  *)
    echo "未知的数组任务号 ${SLURM_ARRAY_TASK_ID:-}，本脚本只有 1-2" >&2
    exit 1
    ;;
esac

echo "任务 ${SLURM_ARRAY_TASK_ID:-1} 完成，run_id=${GENDER_DOMAIN_RUN_ID}"

# --------------------------------------------------------------------------
# 本脚本的两个任务与 slurm/run_robustness_light.slurm 的三个任务**全部**结束
# 之后，在登录节点跑综合层（它只读五张结果表，几秒钟）：
#   python3 -m gender_domain.robustness.synthesis build --year=2020
# 综合层刻意不放进任何一个数组：它必须等五个族都写完才有意义，而数组任务
# 之间没有顺序保证。真要串起来，用
#   --dependency=afterok:<本数组作业 id>:<轻量数组作业 id>
# --------------------------------------------------------------------------
