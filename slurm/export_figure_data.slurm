#!/bin/bash
#SBATCH --job-name=figdata
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --output=job.%j.figdata.out
#SBATCH --mail-type=end
#SBATCH --mail-user=2401111059@stu.pku.edu.cn

# 把服务器上的结果导出成一小撮"能下载"的文件，供本地画 §12 的图。
# 必须排在 run_results.slurm 之后：本步骤只搬运结果表，不重算任何数字，
# 结果表缺一张就直接报错退出（见 export_figure_data.copy_result_tables）。
#
# Output: analysis_data/figure_data/
#           12 张结果表（与 analysis_data/results/ 同名同内容）
#           combo_distribution.parquet      §8 四类来源组合的观测性别分布
#           dist_retweet_count.parquet      附录 ECDF：来源转发次数（抽样）
#           dist_delay_hours.parquet        附录 ECDF：转发延迟（抽样）
#           dist_char_density.parquet       附录 ECDF：字符 density（抽样）
#           distribution_summary.parquet    每格的全量样本量、缺失数、抽样数、
#                                           seed、上限与全量/抽样分位数
#         analysis_data/results/manifest_figures/manifest.json
#
# 内存：延迟分布要把表 B 的清理结果读进内存（复用
# models_temporal.load_clean_events 的同一套清理规则，绝不另写一遍），
# 因此内存按 models_temporal 的量级要 64G；只跑一个阈值口径（720 小时），
# 比 models_temporal 少一份驻留。
# 抽样：每个 性别 × 领域 格子最多 5 万点，seed 固定为 20200101 并写进
# manifest 与 distribution_summary.parquet，换台机器也能抽出逐点相同的样本。
# 可用 --max_points / --seed 覆盖。
#
# Usage:
#   sbatch slurm/export_figure_data.slurm
#
# 跑完之后把 analysis_data/figure_data/ 整个目录下载到本地，然后：
#   python3 -m gender_domain.figures all --year=2020

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opinion

python -m gender_domain.export_figure_data build --year=2020
