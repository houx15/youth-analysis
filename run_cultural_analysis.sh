#!/bin/bash
#SBATCH -o job.%j.cultural_analysis.out
#SBATCH -p C032M0128G
#SBATCH --qos=low
#SBATCH -J cultural_analysis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
# 调度参数沿用用户已验证可运行的北大集群模板。
# 内容行为及语义分析独立作业。
# 用法：sbatch run_cultural_analysis.sh build --input-dir cleaned_weibo_cov/2020 --vocabulary /path/to/vocab.json --output /path/to/new_run
module load anaconda/3.11
source ~/.bash_profile
conda activate opinion
python -m cultural_participation.research "$@"
