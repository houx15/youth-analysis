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
# tables (表 1/表 2). Extend this script in place as later tasks (models,
# temporal analysis) land, appending their build steps after describe.
# Output: analysis_data/user_domain_2020_with_profile.parquet,
#         analysis_data/results/table1_sample_activity.parquet,
#         analysis_data/results/table2_raw_gender_gaps.parquet
#
# Usage:
#   sbatch slurm/run_results.slurm

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opinion

python -m gender_domain.profile_join build --year=2020
python -m gender_domain.describe build --year=2020
