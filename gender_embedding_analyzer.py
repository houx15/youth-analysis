"""
性别和职业词的embedding分析器

功能：
1. 加载已训练的Word2Vec模型
2. 计算职业词与性别词的关联度（分别计算与男性词、女性词的相似度）
3. 比较不同省份模型的差异
4. 生成分析报告和可视化数据

输入：embedding_models/{year}/ 下的模型文件
输出：embedding_analysis/{year}/ 下的分析结果
"""

import os
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import fire
from sklearn.preprocessing import normalize
import warnings
import json
import glob

warnings.filterwarnings("ignore")

MODEL_DIR = "embedding_models"
OUTPUT_DIR = "embedding_analysis"

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 性别词表（扩展版）
GENDER_WORDS = {
    "male": [
        # 代词
        "他",
        "他们",
        "他的",
        # 基础性别词
        "男",
        "男人",
        "男性",
        "男子",
        "男生",
        "男孩",
        # 称谓
        "先生",
        "帅哥",
        "小伙",
        "小伙子",
        "哥",
        "兄弟",
        "爷们",
        # 家庭角色
        "父亲",
        "爸爸",
        "爸",
        "儿子",
        "丈夫",
        "老公",
        "男友",
        "男朋友",
    ],
    "female": [
        # 代词
        "她",
        "她们",
        "她的",
        # 基础性别词
        "女",
        "女人",
        "女性",
        "女子",
        "女生",
        "女孩",
        # 称谓
        "女士",
        "小姐",
        "美女",
        "姑娘",
        "小姑娘",
        "姐",
        "妹",
        "姐妹",
        "闺蜜",
        # 家庭角色
        "母亲",
        "妈妈",
        "妈",
        "女儿",
        "闺女",
        "妻子",
        "老婆",
        "女友",
        "女朋友",
    ],
}

# 职业词表（扩展版，按预期性别刻板程度分类）
OCCUPATION_WORDS = {
    # 预期偏女性的职业
    "female_stereotyped": [
        "护士",
        "幼师",
        "幼儿教师",
        "保姆",
        "月嫂",
        "秘书",
        "前台",
        "文员",
        "客服",
        "收银员",
        "导购",
        "美容师",
        "化妆师",
        "空姐",
        "模特",
        "瑜伽教练",
    ],
    # 预期偏男性的职业
    "male_stereotyped": [
        "程序员",
        "工程师",
        "司机",
        "厨师",
        "保安",
        "建筑工",
        "快递员",
        "外卖员",
        "电工",
        "机械师",
        "军人",
        "警察",
        "消防员",
        "飞行员",
        "船员",
    ],
    # 预期相对中性的职业
    "neutral": [
        "教师",
        "老师",
        "医生",
        "会计",
        "律师",
        "记者",
        "设计师",
        "翻译",
        "作家",
        "演员",
        "歌手",
        "经理",
        "销售",
        "公务员",
        "职员",
    ],
    # 高地位职业
    "high_status": [
        "老板",
        "总裁",
        "董事长",
        "CEO",
        "院长",
        "校长",
        "教授",
        "科学家",
        "研究员",
        "专家",
        "博士",
    ],
}

# 合并所有职业词
ALL_OCCUPATIONS = []
for category in OCCUPATION_WORDS.values():
    ALL_OCCUPATIONS.extend(category)


def get_word_embedding(model, word):
    """获取词向量"""
    try:
        return model.wv[word]
    except KeyError:
        return None


def get_word_set_embedding(model, words):
    """获取一组词的平均向量（归一化）"""
    vectors = []
    found_words = []

    for word in words:
        vec = get_word_embedding(model, word)
        if vec is not None:
            vectors.append(vec)
            found_words.append(word)

    if not vectors:
        return None, []

    # 计算平均向量并归一化
    avg_vec = np.mean(vectors, axis=0)
    normalized_vec = normalize([avg_vec])[0]

    return normalized_vec, found_words


def cosine_similarity(vec1, vec2):
    """计算余弦相似度"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def compute_gender_bias(occupation_vec, male_vec, female_vec):
    """
    计算职业的性别偏向分数

    返回：
        bias_score: 正值=偏女性，负值=偏男性，接近0=中性
        male_sim: 与男性词的相似度
        female_sim: 与女性词的相似度
    """
    male_sim = cosine_similarity(occupation_vec, male_vec)
    female_sim = cosine_similarity(occupation_vec, female_vec)

    # 性别偏向分数 = 女性相似度 - 男性相似度
    bias_score = female_sim - male_sim

    return bias_score, male_sim, female_sim


def load_models(year, province_filter=None):
    """加载指定年份的所有模型"""
    year_model_dir = os.path.join(MODEL_DIR, str(year))
    if not os.path.exists(year_model_dir):
        print(f"❌ 未找到 {year} 年的模型目录: {year_model_dir}")
        return {}

    pattern = os.path.join(year_model_dir, "model_*.model")
    model_files = sorted(glob.glob(pattern))

    if not model_files:
        print(f"❌ 未找到 {year} 年的模型文件")
        return {}

    print(f"📂 找到 {len(model_files)} 个模型文件")

    models = {}
    for model_path in model_files:
        # 从文件名提取省份名称
        filename = os.path.basename(model_path)
        province = filename.replace("model_", "").replace(".model", "")

        # 如果指定了省份过滤，只加载该省份
        if province_filter and province != province_filter:
            continue

        try:
            model = Word2Vec.load(model_path)
            models[province] = model
            print(f"  ✓ 已加载: {province} (词汇量: {len(model.wv):,})")
        except Exception as e:
            print(f"  ❌ 加载失败: {province} - {e}")

    return models


def analyze_model(province, model):
    """分析单个省份的模型"""
    print(f"\n{'='*60}")
    print(f"🔍 分析省份: {province}")
    print(f"{'='*60}")

    vocab_size = len(model.wv)
    print(f"  📊 词汇表大小: {vocab_size:,}")

    # 计算性别词向量
    male_vec, male_found = get_word_set_embedding(model, GENDER_WORDS["male"])
    female_vec, female_found = get_word_set_embedding(model, GENDER_WORDS["female"])

    if male_vec is None or female_vec is None:
        print(f"  ❌ 性别词向量计算失败")
        return None

    print(f"  ✓ 找到男性词: {len(male_found)}/{len(GENDER_WORDS['male'])} 个")
    print(f"    {', '.join(male_found[:10])}{'...' if len(male_found) > 10 else ''}")
    print(f"  ✓ 找到女性词: {len(female_found)}/{len(GENDER_WORDS['female'])} 个")
    print(f"    {', '.join(female_found[:10])}{'...' if len(female_found) > 10 else ''}")

    # 计算每个职业词的性别偏向
    occupation_results = []
    found_occupations = []

    for occupation in ALL_OCCUPATIONS:
        occ_vec = get_word_embedding(model, occupation)
        if occ_vec is not None:
            bias_score, male_sim, female_sim = compute_gender_bias(
                occ_vec, male_vec, female_vec
            )

            occupation_results.append(
                {
                    "occupation": occupation,
                    "bias_score": float(bias_score),
                    "male_similarity": float(male_sim),
                    "female_similarity": float(female_sim),
                }
            )
            found_occupations.append(occupation)

    if not occupation_results:
        print(f"  ❌ 没有找到任何职业词")
        return None

    print(f"  ✓ 找到职业词: {len(found_occupations)}/{len(ALL_OCCUPATIONS)} 个")

    # 排序并展示结果
    occupation_results_sorted = sorted(
        occupation_results, key=lambda x: x["bias_score"], reverse=True
    )

    print(f"\n  📊 职业性别偏向分析:")
    print(f"\n  🔵 最偏女性的职业 (Top 5):")
    for i, occ in enumerate(occupation_results_sorted[:5], 1):
        print(
            f"    {i}. {occ['occupation']:8s} | 偏向分数: {occ['bias_score']:+.3f} "
            f"| 女性相似度: {occ['female_similarity']:.3f} "
            f"| 男性相似度: {occ['male_similarity']:.3f}"
        )

    print(f"\n  🔴 最偏男性的职业 (Top 5):")
    for i, occ in enumerate(occupation_results_sorted[-5:][::-1], 1):
        print(
            f"    {i}. {occ['occupation']:8s} | 偏向分数: {occ['bias_score']:+.3f} "
            f"| 女性相似度: {occ['female_similarity']:.3f} "
            f"| 男性相似度: {occ['male_similarity']:.3f}"
        )

    # 计算统计指标
    bias_scores = [r["bias_score"] for r in occupation_results]
    stats = {
        "province": province,
        "vocab_size": vocab_size,
        "occupations_found": len(found_occupations),
        "male_words_found": len(male_found),
        "female_words_found": len(female_found),
        "mean_bias": float(np.mean(bias_scores)),
        "std_bias": float(np.std(bias_scores)),
        "min_bias": float(np.min(bias_scores)),
        "max_bias": float(np.max(bias_scores)),
        "range_bias": float(np.max(bias_scores) - np.min(bias_scores)),
    }

    print(f"\n  📈 统计指标:")
    print(f"    平均偏向: {stats['mean_bias']:+.3f}")
    print(f"    标准差（隔离程度）: {stats['std_bias']:.3f}")
    print(f"    偏向范围: [{stats['min_bias']:+.3f}, {stats['max_bias']:+.3f}]")

    # 返回分析结果
    result = {
        "province": province,
        "stats": stats,
        "male_vec": male_vec.tolist(),
        "female_vec": female_vec.tolist(),
        "male_words_found": male_found,
        "female_words_found": female_found,
        "occupations_found": found_occupations,
        "occupation_results": occupation_results,
    }

    return result


def analyze_all_models(models):
    """分析所有省份的模型"""
    results = []
    province_stats = []

    for province, model in models.items():
        result = analyze_model(province, model)
        if result:
            results.append(result)
            province_stats.append(result["stats"])

    return results, province_stats


def save_results(results, province_stats, year):
    """保存分析结果"""
    if not results:
        print("❌ 没有生成任何结果")
        return

    year_output_dir = os.path.join(OUTPUT_DIR, str(year))
    os.makedirs(year_output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"💾 保存结果...")
    print(f"{'='*60}")

    # 1. 保存省份统计信息
    stats_df = pd.DataFrame(province_stats)
    stats_file = os.path.join(year_output_dir, f"province_stats.csv")
    stats_df.to_csv(stats_file, index=False, encoding="utf-8-sig")
    print(f"✓ 省份统计信息: {stats_file}")

    # 2. 保存职业性别偏向详细数据（长格式）
    occupation_data = []
    for result in results:
        province = result["province"]
        for occ in result["occupation_results"]:
            occupation_data.append(
                {
                    "province": province,
                    "occupation": occ["occupation"],
                    "bias_score": occ["bias_score"],
                    "male_similarity": occ["male_similarity"],
                    "female_similarity": occ["female_similarity"],
                }
            )

    occupation_df = pd.DataFrame(occupation_data)
    occupation_file = os.path.join(year_output_dir, f"occupation_bias.csv")
    occupation_df.to_csv(occupation_file, index=False, encoding="utf-8-sig")
    print(f"✓ 职业性别偏向数据: {occupation_file}")

    # 3. 保存宽格式数据（省份×职业矩阵）
    pivot_df = occupation_df.pivot_table(
        values="bias_score", index="occupation", columns="province", aggfunc="mean"
    )
    pivot_file = os.path.join(year_output_dir, f"occupation_bias_pivot.csv")
    pivot_df.to_csv(pivot_file, encoding="utf-8-sig")
    print(f"✓ 职业×省份矩阵: {pivot_file}")

    # 4. 保存详细向量数据（JSON格式）
    detailed_data = []
    for result in results:
        detailed_data.append(
            {
                "province": result["province"],
                "stats": result["stats"],
                "male_vec": result["male_vec"],
                "female_vec": result["female_vec"],
                "male_words_found": result["male_words_found"],
                "female_words_found": result["female_words_found"],
                "occupations_found": result["occupations_found"],
            }
        )

    detailed_file = os.path.join(year_output_dir, f"detailed_vectors.json")
    with open(detailed_file, "w", encoding="utf-8") as f:
        json.dump(detailed_data, f, ensure_ascii=False, indent=2)
    print(f"✓ 详细向量数据: {detailed_file}")

    # 5. 生成简要分析报告
    report_file = os.path.join(year_output_dir, f"analysis_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"{'='*60}\n")
        f.write(f"性别-职业Embedding分析报告 ({year}年)\n")
        f.write(f"{'='*60}\n\n")

        f.write(f"分析省份数: {len(results)}\n")
        f.write(f"分析职业数: {len(ALL_OCCUPATIONS)}\n\n")

        f.write(f"{'='*60}\n")
        f.write(f"各省份性别隔离指数排名（标准差）:\n")
        f.write(f"{'='*60}\n")
        stats_sorted = sorted(province_stats, key=lambda x: x["std_bias"], reverse=True)
        for i, stat in enumerate(stats_sorted, 1):
            f.write(
                f"{i:2d}. {stat['province']:10s} | "
                f"隔离指数: {stat['std_bias']:.3f} | "
                f"平均偏向: {stat['mean_bias']:+.3f}\n"
            )

        f.write(f"\n{'='*60}\n")
        f.write(f"职业性别偏向一致性分析:\n")
        f.write(f"{'='*60}\n")

        # 计算每个职业在各省份的平均偏向
        occupation_avg = (
            occupation_df.groupby("occupation")["bias_score"]
            .agg(["mean", "std"])
            .sort_values("mean", ascending=False)
        )

        f.write(f"\n最偏女性的职业（跨省份平均）:\n")
        for i, (occ, row) in enumerate(occupation_avg.head(10).iterrows(), 1):
            f.write(
                f"  {i:2d}. {occ:15s} | 平均: {row['mean']:+.3f} | 标准差: {row['std']:.3f}\n"
            )

        f.write(f"\n最偏男性的职业（跨省份平均）:\n")
        for i, (occ, row) in enumerate(
            occupation_avg.tail(10).iloc[::-1].iterrows(), 1
        ):
            f.write(
                f"  {i:2d}. {occ:15s} | 平均: {row['mean']:+.3f} | 标准差: {row['std']:.3f}\n"
            )

        f.write(f"\n职业偏向差异最大的（跨省份标准差最大）:\n")
        occupation_var = occupation_avg.sort_values("std", ascending=False)
        for i, (occ, row) in enumerate(occupation_var.head(10).iterrows(), 1):
            f.write(
                f"  {i:2d}. {occ:15s} | 平均: {row['mean']:+.3f} | 标准差: {row['std']:.3f}\n"
            )

    print(f"✓ 分析报告: {report_file}")

    print(f"\n✅ 所有结果已保存到 {year_output_dir}/ 目录")


def main(year: int, province: str = None):
    """
    运行embedding分析

    Args:
        year: 年份
        province: 指定省份（可选），如果不指定则分析所有省份
    """
    print(f"\n{'='*60}")
    print(f"🚀 开始分析 {year} 年数据的性别-职业Embedding")
    print(f"{'='*60}\n")

    # 加载模型
    models = load_models(year, province)
    if not models:
        print("❌ 无法加载模型")
        return

    # 分析模型
    results, province_stats = analyze_all_models(models)

    # 保存结果
    save_results(results, province_stats, year)

    print(f"\n{'='*60}")
    print(f"🎉 {year} 年embedding分析完成！")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    fire.Fire(main)
