#!/bin/bash
#SBATCH --job-name=results
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=192G
#SBATCH --output=job.%j.results.out
#SBATCH --mail-type=end
#SBATCH --mail-user=2401111059@stu.pku.edu.cn

# Server-side results layer, run in dependency order: profile join first
# (Table C -> user_domain_2020_with_profile.parquet), then the descriptive
# tables (表 1/表 2), then the core models (§6.2–6.5). Extend this script in
# place as later tasks (interaction, temporal analysis) land, appending their
# build steps after models_core.
# Output: analysis_data/user_domain_2020_with_profile.parquet,
#         analysis_data/results/table1_sample_activity.parquet,
#         analysis_data/results/table2_raw_gender_gaps.parquet,
#         analysis_data/results/models_entry.parquet,
#         analysis_data/results/models_intensity.parquet,
#         analysis_data/results/models_share.parquet,
#         analysis_data/results/models_persistence.parquet,
#         analysis_data/results/interaction_gender_domain.parquet,
#         analysis_data/results/decomposition_source_content.parquet,
#         analysis_data/results/combination_multinomial.parquet,
#         analysis_data/results/monthly_rates.parquet,
#         analysis_data/results/leave_one_month_out.parquet,
#         analysis_data/results/delay_quantiles.parquet
#
# models_core depends on profile_join: M2 只在 profile_complete 的用户上拟合，
# 没有跑过 profile_join 就没有这一列，M2 会整层退化成失败留痕行。
# models_interaction 同样依赖 profile_join（M2 层），并且是论文的核心结果
# （§6.6 Gender × Domain 差中差），排在 models_core 之后运行。
# models_combination（§7 来源/内容分解、§8 参与组合）同样依赖 profile_join
# 的 M2 列，排在 models_interaction 之后：它与核心结果没有依赖关系，但一旦
# 作业撞上墙钟被杀，先跑完核心结果更划算。它自己按阶段增量落盘，中途被杀
# 时已经算完的阶段仍留在磁盘上（见模块里的 _write_partial）。
# models_temporal（§9 月度稳定性、§10 转发延迟）排在最后：它读的是表 D
# （用户—月份面板）与表 B（转发事件表），与前面的模块没有依赖关系，但它是
# 本层里最贵的一步——26 次留一月重估（两个领域各 13 次）都在两百多万行的
# 面板上拟合、外加按用户聚类的标准误，再加上事件表上的簇 bootstrap。墙钟
# 因此从 4 小时提到 8 小时；它同样按阶段增量落盘，被杀时已算完的阶段仍在盘上。
# 延迟阶段的 bootstrap 次数可以用 --n_boot 调低（区间会变粗，但仍然是按用户
# 整簇重抽样的区间）。延迟结果默认在两个超长延迟阈值口径上各跑一遍
# （720 小时与不设上限），并列写进 delay_quantiles.parquet 的不同 model 变体，
# 让阈值这条裁定的代价直接可读；口径可用 --delay_thresholds 覆盖。
#
# 资源额度（M18）：statsmodels 本身是单线程的，cpus-per-task 只对底层 BLAS
# （零截断负二项与 GEE 的矩阵运算）有用，因此 4 -> 8 只是给 BLAS 一点余量，
# 不指望线性加速。真正的瓶颈是墙钟与内存：最贵的一步是 §9 的 26 次留一月
# 重估（两个域各 13 次），每一次都在两百多万行的用户—月面板上拟合并算按
# 用户聚类的标准误；再加上 §10 的簇 bootstrap 与 §7/§8 的 bootstrap。原来
# 的 8 小时 / 64G 没有任何余量——一旦撞墙钟被杀，盘上留下的就是一组半新
# 半旧的结果表（正是下面 set -e 与运行标识要防的那种局面），而重跑一次的
# 代价远高于多申请的这点额度。因此提到 24 小时 / 192G。
#
# 失败处理（C1）：set -euo pipefail 必须有。此前六条命令顺序执行、没有任何
# 错误处理，models_core 崩掉之后后面几步照常运行，作业的退出码是最后一条
# 命令的退出码，--mail-type=end 无论如何都报告"结束"。于是盘上留下
# "describe 是本次的、models_entry 是上一次的"这种组合，而导出层当时只检查
# 文件在不在——旧文件同样在。现在两道闸门一起上：
#   1) set -euo pipefail：任何一步非零退出，作业立刻停下，退出码传出去；
#   2) GENDER_DOMAIN_RUN_ID：六个步骤共用同一个运行标识，各自把它记进
#      analysis_data/results/run_stamps.json；导出层
#      （export_figure_data.copy_result_tables -> config.verify_same_run）
#      逐张核对，只要有一张来自别的批次就报错并点名。
# 刻意**不**在开头 rm -rf analysis_data/results/：models_combination 与
# models_temporal 是按阶段增量落盘的（见各模块的 _write_partial），撞墙钟
# 被杀时已算完的阶段还留在盘上是有意为之的设计。清空目录会把这个设计一起
# 抹掉，而"新旧混装"这件事已经由运行标识挡住了——那才是真正的闸门，
# 清目录只是个补充。
#
# Usage:
#   sbatch slurm/run_results.slurm

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opinion

# 同一次作业的六个步骤共用一个运行标识。用 SLURM_JOB_ID 是因为它天然唯一
# 且能和 job.%j.results.out 的日志对上；手工在登录节点跑时退化成时间戳。
export GENDER_DOMAIN_RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d%H%M%S)}"
echo "GENDER_DOMAIN_RUN_ID=${GENDER_DOMAIN_RUN_ID}"

python -m gender_domain.profile_join build --year=2020
python -m gender_domain.describe build --year=2020
python -m gender_domain.models_core build --year=2020
python -m gender_domain.models_interaction build --year=2020
python -m gender_domain.models_combination build --year=2020
python -m gender_domain.models_temporal build --year=2020

echo "全部六个步骤完成，run_id=${GENDER_DOMAIN_RUN_ID}"
