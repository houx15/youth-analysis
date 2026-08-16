#!/bin/bash
#SBATCH --job-name=results
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --mem=64G
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
# Usage:
#   sbatch slurm/run_results.slurm

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opinion

python -m gender_domain.profile_join build --year=2020
python -m gender_domain.describe build --year=2020
python -m gender_domain.models_core build --year=2020
python -m gender_domain.models_interaction build --year=2020
python -m gender_domain.models_combination build --year=2020
python -m gender_domain.models_temporal build --year=2020
