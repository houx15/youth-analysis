"""
性别和职业词的embedding分析器

功能：
1. 加载已训练的Word2Vec模型
2. 计算职业词与性别词的关联度（分别计算与男性词、女性词的相似度）
3. 比较不同省份模型的差异
4. 分析家务分工词汇（家庭场域 vs 工作场域）的性别差异
5. 生成分析报告和可视化数据

输入：gender_embedding/embedding_models/{year}/ 下的模型文件
输出：embedding_analysis/{year}/ 下的分析结果
"""

import os
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
import fire
from sklearn.preprocessing import normalize
import warnings
import json
import glob

warnings.filterwarnings("ignore")

MODEL_DIR = "gender_embedding/embedding_models"
OUTPUT_DIR = "embedding_analysis"
WORDLISTS_DIR = "wordlists"

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(WORDLISTS_DIR, exist_ok=True)


def load_json_wordlist(filename):
    """
    从JSON文件加载词表

    Args:
        filename: JSON词表文件名（在wordlists目录下）

    Returns:
        dict: 词表字典
    """
    filepath = os.path.join(WORDLISTS_DIR, filename)
    if not os.path.exists(filepath):
        print(f"⚠️  词表文件不存在: {filepath}")
        return {}

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ 加载词表文件失败 {filepath}: {e}")
        return {}


def load_gender_words():
    """加载性别词表"""
    data = load_json_wordlist("gender_words.json")
    return {"male": data.get("male", []), "female": data.get("female", [])}


def load_occupation_words():
    """加载职业词表，返回所有职业词的列表"""
    data = load_json_wordlist("occupation_words.json")
    all_occupations = []
    for category in data.values():
        all_occupations.extend(category)
    return all_occupations


def load_domestic_work_words():
    """加载家务分工词表"""
    data = load_json_wordlist("domestic_work_words.json")
    return {"family": data.get("family", []), "work": data.get("work", [])}


# 加载词表
GENDER_WORDS = load_gender_words()
ALL_OCCUPATIONS = load_occupation_words()
DOMESTIC_WORK_WORDS = load_domestic_work_words()


def get_word_embedding(model, word):
    """获取词向量"""
    try:
        return model[word]
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
    计算职业的性别偏向分数（基于余弦相似度差值）

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


def compute_gender_bias_projection(occupation_vec, male_vec, female_vec):
    """
    计算职业的性别偏向分数（基于性别轴投影）

    构建性别轴：从男性向量指向女性向量的方向向量（正向为女性，负向为男性）
    计算职业词向量在性别轴上的投影值

    返回：
        projection_score: 投影值，正值=偏女性，负值=偏男性，接近0=中性
        gender_axis: 性别轴方向向量（归一化）
    """
    # 构建性别轴：女性向量 - 男性向量（正向为女性方向）
    gender_axis = female_vec - male_vec

    # 归一化性别轴
    axis_norm = np.linalg.norm(gender_axis)
    if axis_norm > 0:
        gender_axis_normalized = gender_axis / axis_norm
    else:
        # 如果性别轴为零向量，返回0
        return 0.0, gender_axis

    # 计算职业词向量在性别轴上的投影
    # projection = dot(occupation_vec, gender_axis_normalized)
    projection_score = np.dot(occupation_vec, gender_axis_normalized)

    return projection_score, gender_axis_normalized


def compute_domain_bias(word_vec, family_vec, work_vec):
    """
    计算词汇在家庭场域 vs 工作场域的偏向分数

    返回：
        bias_score: 正值=偏家庭场域，负值=偏工作场域，接近0=中性
        family_sim: 与家庭场域词的相似度
        work_sim: 与工作场域词的相似度
    """
    family_sim = cosine_similarity(word_vec, family_vec)
    work_sim = cosine_similarity(word_vec, work_vec)

    # 场域偏向分数 = 家庭相似度 - 工作相似度
    bias_score = family_sim - work_sim

    return bias_score, family_sim, work_sim


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
            model = KeyedVectors.load(model_path)
            models[province] = model
            print(f"  ✓ 已加载: {province} (词汇量: {len(model):,})")
        except Exception as e:
            print(f"  ❌ 加载失败: {province} - {e}")

    return models


def analyze_model(province, model):
    """分析单个省份的模型"""
    print(f"\n{'='*60}")
    print(f"🔍 分析省份: {province}")
    print(f"{'='*60}")

    vocab_size = len(model)
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
    print(
        f"    {', '.join(female_found[:10])}{'...' if len(female_found) > 10 else ''}"
    )

    # 计算每个职业词的性别偏向（使用两种方法）
    occupation_results = []
    found_occupations = []

    for occupation in ALL_OCCUPATIONS:
        occ_vec = get_word_embedding(model, occupation)
        if occ_vec is not None:
            # 方法1：余弦相似度差值
            bias_score, male_sim, female_sim = compute_gender_bias(
                occ_vec, male_vec, female_vec
            )

            # 方法2：性别轴投影
            projection_score, _ = compute_gender_bias_projection(
                occ_vec, male_vec, female_vec
            )

            occupation_results.append(
                {
                    "occupation": occupation,
                    "bias_score": float(bias_score),  # 余弦相似度差值方法
                    "projection_score": float(projection_score),  # 性别轴投影方法
                    "male_similarity": float(male_sim),
                    "female_similarity": float(female_sim),
                }
            )
            found_occupations.append(occupation)

    if not occupation_results:
        print(f"  ❌ 没有找到任何职业词")
        return None

    print(f"  ✓ 找到职业词: {len(found_occupations)}/{len(ALL_OCCUPATIONS)} 个")

    # 排序并展示结果（按余弦相似度差值）
    occupation_results_sorted = sorted(
        occupation_results, key=lambda x: x["bias_score"], reverse=True
    )

    print(f"\n  📊 职业性别偏向分析（余弦相似度差值方法）:")
    print(f"\n  🔵 最偏女性的职业 (Top 5):")
    for i, occ in enumerate(occupation_results_sorted[:5], 1):
        print(
            f"    {i}. {occ['occupation']:8s} | 偏向分数: {occ['bias_score']:+.3f} "
            f"| 投影分数: {occ['projection_score']:+.3f} "
            f"| 女性相似度: {occ['female_similarity']:.3f} "
            f"| 男性相似度: {occ['male_similarity']:.3f}"
        )

    print(f"\n  🔴 最偏男性的职业 (Top 5):")
    for i, occ in enumerate(occupation_results_sorted[-5:][::-1], 1):
        print(
            f"    {i}. {occ['occupation']:8s} | 偏向分数: {occ['bias_score']:+.3f} "
            f"| 投影分数: {occ['projection_score']:+.3f} "
            f"| 女性相似度: {occ['female_similarity']:.3f} "
            f"| 男性相似度: {occ['male_similarity']:.3f}"
        )

    # 按投影分数排序并展示
    occupation_results_sorted_proj = sorted(
        occupation_results, key=lambda x: x["projection_score"], reverse=True
    )

    print(f"\n  📊 职业性别偏向分析（性别轴投影方法）:")
    print(f"\n  🔵 最偏女性的职业 (Top 5):")
    for i, occ in enumerate(occupation_results_sorted_proj[:5], 1):
        print(
            f"    {i}. {occ['occupation']:8s} | 投影分数: {occ['projection_score']:+.3f} "
            f"| 偏向分数: {occ['bias_score']:+.3f}"
        )

    print(f"\n  🔴 最偏男性的职业 (Top 5):")
    for i, occ in enumerate(occupation_results_sorted_proj[-5:][::-1], 1):
        print(
            f"    {i}. {occ['occupation']:8s} | 投影分数: {occ['projection_score']:+.3f} "
            f"| 偏向分数: {occ['bias_score']:+.3f}"
        )

    # 计算家务分工词汇分析
    family_vec, family_found = get_word_set_embedding(
        model, DOMESTIC_WORK_WORDS["family"]
    )
    work_vec, work_found = get_word_set_embedding(model, DOMESTIC_WORK_WORDS["work"])

    domestic_work_results = []
    if family_vec is not None and work_vec is not None:
        print(f"\n  📊 家务分工场域分析:")
        print(
            f"  ✓ 找到家庭场域词: {len(family_found)}/{len(DOMESTIC_WORK_WORDS['family'])} 个"
        )
        print(
            f"  ✓ 找到工作场域词: {len(work_found)}/{len(DOMESTIC_WORK_WORDS['work'])} 个"
        )

        # 计算男性词和女性词在家庭场域 vs 工作场域的偏向
        male_domain_bias, male_family_sim, male_work_sim = compute_domain_bias(
            male_vec, family_vec, work_vec
        )
        female_domain_bias, female_family_sim, female_work_sim = compute_domain_bias(
            female_vec, family_vec, work_vec
        )

        domestic_work_results.append(
            {
                "word_type": "male",
                "domain_bias": float(male_domain_bias),
                "family_similarity": float(male_family_sim),
                "work_similarity": float(male_work_sim),
            }
        )
        domestic_work_results.append(
            {
                "word_type": "female",
                "domain_bias": float(female_domain_bias),
                "family_similarity": float(female_family_sim),
                "work_similarity": float(female_work_sim),
            }
        )

        print(f"\n  🏠 性别在家庭场域 vs 工作场域的偏向:")
        print(
            f"    男性: 场域偏向分数 {male_domain_bias:+.3f} "
            f"(家庭: {male_family_sim:.3f}, 工作: {male_work_sim:.3f})"
        )
        print(
            f"    女性: 场域偏向分数 {female_domain_bias:+.3f} "
            f"(家庭: {female_family_sim:.3f}, 工作: {female_work_sim:.3f})"
        )
        print(
            f"    性别差异: {female_domain_bias - male_domain_bias:+.3f} "
            f"(正值表示女性更偏向家庭场域)"
        )

    # 计算统计指标
    bias_scores = [r["bias_score"] for r in occupation_results]
    projection_scores = [r["projection_score"] for r in occupation_results]
    stats = {
        "province": province,
        "vocab_size": vocab_size,
        "occupations_found": len(found_occupations),
        "male_words_found": len(male_found),
        "female_words_found": len(female_found),
        # 余弦相似度差值方法的统计
        "mean_bias": float(np.mean(bias_scores)),
        "std_bias": float(np.std(bias_scores)),
        "min_bias": float(np.min(bias_scores)),
        "max_bias": float(np.max(bias_scores)),
        "range_bias": float(np.max(bias_scores) - np.min(bias_scores)),
        # 性别轴投影方法的统计
        "mean_projection": float(np.mean(projection_scores)),
        "std_projection": float(np.std(projection_scores)),
        "min_projection": float(np.min(projection_scores)),
        "max_projection": float(np.max(projection_scores)),
        "range_projection": float(
            np.max(projection_scores) - np.min(projection_scores)
        ),
    }

    if domestic_work_results:
        male_domain = next(
            (r for r in domestic_work_results if r["word_type"] == "male"), None
        )
        female_domain = next(
            (r for r in domestic_work_results if r["word_type"] == "female"), None
        )
        if male_domain and female_domain:
            stats["male_domain_bias"] = male_domain["domain_bias"]
            stats["female_domain_bias"] = female_domain["domain_bias"]
            stats["gender_domain_gap"] = (
                female_domain["domain_bias"] - male_domain["domain_bias"]
            )

    print(f"\n  📈 统计指标:")
    print(f"    余弦相似度差值方法:")
    print(f"      平均偏向: {stats['mean_bias']:+.3f}")
    print(f"      标准差（隔离程度）: {stats['std_bias']:.3f}")
    print(f"      偏向范围: [{stats['min_bias']:+.3f}, {stats['max_bias']:+.3f}]")
    print(f"    性别轴投影方法:")
    print(f"      平均投影: {stats['mean_projection']:+.3f}")
    print(f"      标准差: {stats['std_projection']:.3f}")
    print(
        f"      投影范围: [{stats['min_projection']:+.3f}, {stats['max_projection']:+.3f}]"
    )

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
        "domestic_work_results": domestic_work_results,
        "family_words_found": family_found if family_vec is not None else [],
        "work_words_found": work_found if work_vec is not None else [],
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
                    "bias_score": occ["bias_score"],  # 余弦相似度差值方法
                    "projection_score": occ["projection_score"],  # 性别轴投影方法
                    "male_similarity": occ["male_similarity"],
                    "female_similarity": occ["female_similarity"],
                }
            )

    occupation_df = pd.DataFrame(occupation_data)
    occupation_file = os.path.join(year_output_dir, f"occupation_bias.csv")
    occupation_df.to_csv(occupation_file, index=False, encoding="utf-8-sig")
    print(f"✓ 职业性别偏向数据: {occupation_file}")

    # 2.5. 保存家务分工场域偏向数据
    domestic_work_data = []
    for result in results:
        province = result["province"]
        if result.get("domestic_work_results"):
            for dw in result["domestic_work_results"]:
                domestic_work_data.append(
                    {
                        "province": province,
                        "word_type": dw["word_type"],
                        "domain_bias": dw["domain_bias"],
                        "family_similarity": dw["family_similarity"],
                        "work_similarity": dw["work_similarity"],
                    }
                )

    if domestic_work_data:
        domestic_work_df = pd.DataFrame(domestic_work_data)
        domestic_work_file = os.path.join(year_output_dir, f"domestic_work_bias.csv")
        domestic_work_df.to_csv(domestic_work_file, index=False, encoding="utf-8-sig")
        print(f"✓ 家务分工场域偏向数据: {domestic_work_file}")

    # 3. 保存宽格式数据（省份×职业矩阵）
    # 3.1 余弦相似度差值方法的矩阵
    pivot_df = occupation_df.pivot_table(
        values="bias_score", index="occupation", columns="province", aggfunc="mean"
    )
    pivot_file = os.path.join(year_output_dir, f"occupation_bias_pivot.csv")
    pivot_df.to_csv(pivot_file, encoding="utf-8-sig")
    print(f"✓ 职业×省份矩阵（余弦相似度差值）: {pivot_file}")

    # 3.2 性别轴投影方法的矩阵
    pivot_proj_df = occupation_df.pivot_table(
        values="projection_score",
        index="occupation",
        columns="province",
        aggfunc="mean",
    )
    pivot_proj_file = os.path.join(year_output_dir, f"occupation_projection_pivot.csv")
    pivot_proj_df.to_csv(pivot_proj_file, encoding="utf-8-sig")
    print(f"✓ 职业×省份矩阵（性别轴投影）: {pivot_proj_file}")

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
        f.write(f"各省份性别隔离指数排名（余弦相似度差值方法，按标准差）:\n")
        f.write(f"{'='*60}\n")
        stats_sorted = sorted(province_stats, key=lambda x: x["std_bias"], reverse=True)
        for i, stat in enumerate(stats_sorted, 1):
            f.write(
                f"{i:2d}. {stat['province']:10s} | "
                f"隔离指数: {stat['std_bias']:.3f} | "
                f"平均偏向: {stat['mean_bias']:+.3f}\n"
            )

        f.write(f"\n{'='*60}\n")
        f.write(f"各省份性别隔离指数排名（性别轴投影方法，按标准差）:\n")
        f.write(f"{'='*60}\n")
        stats_sorted_proj = sorted(
            province_stats, key=lambda x: x.get("std_projection", 0), reverse=True
        )
        for i, stat in enumerate(stats_sorted_proj, 1):
            f.write(
                f"{i:2d}. {stat['province']:10s} | "
                f"隔离指数: {stat.get('std_projection', 0):.3f} | "
                f"平均投影: {stat.get('mean_projection', 0):+.3f}\n"
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

        # 添加性别轴投影方法分析
        f.write(f"\n{'='*60}\n")
        f.write(f"职业性别偏向分析（性别轴投影方法）:\n")
        f.write(f"{'='*60}\n")

        # 计算每个职业在各省份的平均投影分数
        occupation_proj_avg = (
            occupation_df.groupby("occupation")["projection_score"]
            .agg(["mean", "std"])
            .sort_values("mean", ascending=False)
        )

        f.write(f"\n最偏女性的职业（跨省份平均，按投影分数）:\n")
        for i, (occ, row) in enumerate(occupation_proj_avg.head(10).iterrows(), 1):
            f.write(
                f"  {i:2d}. {occ:15s} | 平均投影: {row['mean']:+.3f} | 标准差: {row['std']:.3f}\n"
            )

        f.write(f"\n最偏男性的职业（跨省份平均，按投影分数）:\n")
        for i, (occ, row) in enumerate(
            occupation_proj_avg.tail(10).iloc[::-1].iterrows(), 1
        ):
            f.write(
                f"  {i:2d}. {occ:15s} | 平均投影: {row['mean']:+.3f} | 标准差: {row['std']:.3f}\n"
            )

        f.write(f"\n职业投影差异最大的（跨省份标准差最大）:\n")
        occupation_proj_var = occupation_proj_avg.sort_values("std", ascending=False)
        for i, (occ, row) in enumerate(occupation_proj_var.head(10).iterrows(), 1):
            f.write(
                f"  {i:2d}. {occ:15s} | 平均投影: {row['mean']:+.3f} | 标准差: {row['std']:.3f}\n"
            )

        # 添加家务分工场域分析
        if domestic_work_data:
            f.write(f"\n{'='*60}\n")
            f.write(f"家务分工场域分析（家庭场域 vs 工作场域）:\n")
            f.write(f"{'='*60}\n")

            domestic_work_df = pd.DataFrame(domestic_work_data)

            # 计算每个省份的性别场域差异
            province_domain_gaps = []
            for province in domestic_work_df["province"].unique():
                prov_df = domestic_work_df[domestic_work_df["province"] == province]
                male_bias = prov_df[prov_df["word_type"] == "male"][
                    "domain_bias"
                ].values
                female_bias = prov_df[prov_df["word_type"] == "female"][
                    "domain_bias"
                ].values

                if len(male_bias) > 0 and len(female_bias) > 0:
                    gap = female_bias[0] - male_bias[0]
                    province_domain_gaps.append(
                        {"province": province, "gender_domain_gap": gap}
                    )

            if province_domain_gaps:
                province_gaps_df = pd.DataFrame(province_domain_gaps)
                province_gaps_df = province_gaps_df.sort_values(
                    "gender_domain_gap", ascending=False
                )

                f.write(f"\n各省份性别在家庭场域 vs 工作场域的差异排名:\n")
                f.write(f"(正值表示女性更偏向家庭场域，负值表示男性更偏向家庭场域)\n")
                for i, row in enumerate(province_gaps_df.itertuples(), 1):
                    f.write(
                        f"  {i:2d}. {row.province:10s} | 性别场域差异: {row.gender_domain_gap:+.3f}\n"
                    )

            # 计算跨省份平均
            male_avg = domestic_work_df[domestic_work_df["word_type"] == "male"][
                "domain_bias"
            ].mean()
            female_avg = domestic_work_df[domestic_work_df["word_type"] == "female"][
                "domain_bias"
            ].mean()
            f.write(f"\n跨省份平均场域偏向:\n")
            f.write(f"  男性: {male_avg:+.3f}\n")
            f.write(f"  女性: {female_avg:+.3f}\n")
            f.write(f"  性别差异: {female_avg - male_avg:+.3f}\n")

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
