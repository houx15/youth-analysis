#!/bin/bash
#SBATCH --job-name=results
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
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
#         analysis_data/results/interaction_gender_domain.parquet
#
# models_core depends on profile_join: M2 只在 profile_complete 的用户上拟合，
# 没有跑过 profile_join 就没有这一列，M2 会整层退化成失败留痕行。
# models_interaction 同样依赖 profile_join（M2 层），并且是论文的核心结果
# （§6.6 Gender × Domain 差中差），排在 models_core 之后运行。
#
# Usage:
#   sbatch slurm/run_results.slurm

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opinion

python -m gender_domain.profile_join build --year=2020
python -m gender_domain.describe build --year=2020
python -m gender_domain.models_core build --year=2020
python -m gender_domain.models_interaction build --year=2020
