#!/bin/bash
#SBATCH --job-name=sample_weibo_translation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --mail-type=end
#SBATCH --mail-user=2401111059@stu.pku.edu.cn

# Sample 200 weibos (proportional-stratified by opinion class) with their
# existing Kimi-Chinese sentiment scores, for cross-lingual validation.
# Output: ai_attitudes/weibo_translation_sample.parquet
#
# Usage:
#   sbatch slurm/sample_weibo_translation.slurm

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opinion

python sample_for_translation.py --sample_size 200
