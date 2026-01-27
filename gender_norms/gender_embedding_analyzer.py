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

MODEL_DIR = "gender_norms/gender_embedding/embedding_models"
OUTPUT_DIR = "gender_norms/gender_embedding/results/embedding_analysis"
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
    计算词汇在家庭场域 vs 工作场域的偏向分数（基于余弦相似度差值）

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


def get_available_provinces(year):
    """获取指定年份所有可用的省份列表（不加载模型）"""
    year_model_dir = os.path.join(MODEL_DIR, str(year))
    if not os.path.exists(year_model_dir):
        return []

    pattern = os.path.join(year_model_dir, "model_*.model")
    model_files = sorted(glob.glob(pattern))

    provinces = []
    for model_path in model_files:
        filename = os.path.basename(model_path)
        province = filename.replace("model_", "").replace(".model", "")
        provinces.append(province)

    return sorted(provinces)


def load_single_model(year, province):
    """加载指定年份和省份的单个模型"""
    year_model_dir = os.path.join(MODEL_DIR, str(year))
    model_path = os.path.join(year_model_dir, f"model_{province}.model")

    if not os.path.exists(model_path):
        print(f"❌ 未找到模型文件: {model_path}")
        return None

    try:
        model = KeyedVectors.load(model_path)
        print(f"  ✓ 已加载: {province} (词汇量: {len(model):,})")
        return model
    except Exception as e:
        print(f"  ❌ 加载失败: {province} - {e}")
        return None


def load_models(year, province_filter=None):
    """加载指定年份的所有模型（保留此函数以保持向后兼容）"""
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
    # 只保留简要的进度信息在console
    print(f"  分析省份: {province}")

    # 用于收集详细报告的列表
    report_lines = []

    report_lines.append(f"\n{'='*60}")
    report_lines.append(f"省份: {province}")
    report_lines.append(f"{'='*60}\n")

    vocab_size = len(model)
    report_lines.append(f"词汇表大小: {vocab_size:,}")

    # 计算性别词向量
    male_vec, male_found = get_word_set_embedding(model, GENDER_WORDS["male"])
    female_vec, female_found = get_word_set_embedding(model, GENDER_WORDS["female"])

    if male_vec is None or female_vec is None:
        report_lines.append(f"❌ 性别词向量计算失败\n")
        return None

    report_lines.append(f"找到男性词: {len(male_found)}/{len(GENDER_WORDS['male'])} 个")
    report_lines.append(
        f"  {', '.join(male_found[:15])}{'...' if len(male_found) > 15 else ''}"
    )
    report_lines.append(
        f"找到女性词: {len(female_found)}/{len(GENDER_WORDS['female'])} 个"
    )
    report_lines.append(
        f"  {', '.join(female_found[:15])}{'...' if len(female_found) > 15 else ''}\n"
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
        report_lines.append(f"❌ 没有找到任何职业词\n")
        return None

    report_lines.append(
        f"找到职业词: {len(found_occupations)}/{len(ALL_OCCUPATIONS)} 个\n"
    )

    # 排序并展示结果（按余弦相似度差值）
    occupation_results_sorted = sorted(
        occupation_results, key=lambda x: x["bias_score"], reverse=True
    )

    report_lines.append(f"【职业性别偏向分析 - 余弦相似度差值方法】")
    report_lines.append(f"\n最偏女性的职业 (Top 5):")
    for i, occ in enumerate(occupation_results_sorted[:5], 1):
        report_lines.append(
            f"  {i}. {occ['occupation']:8s} | 偏向分数: {occ['bias_score']:+.3f} "
            f"| 投影分数: {occ['projection_score']:+.3f} "
            f"| 女性相似度: {occ['female_similarity']:.3f} "
            f"| 男性相似度: {occ['male_similarity']:.3f}"
        )

    report_lines.append(f"\n最偏男性的职业 (Top 5):")
    for i, occ in enumerate(occupation_results_sorted[-5:][::-1], 1):
        report_lines.append(
            f"  {i}. {occ['occupation']:8s} | 偏向分数: {occ['bias_score']:+.3f} "
            f"| 投影分数: {occ['projection_score']:+.3f} "
            f"| 女性相似度: {occ['female_similarity']:.3f} "
            f"| 男性相似度: {occ['male_similarity']:.3f}"
        )

    # 按投影分数排序并展示
    occupation_results_sorted_proj = sorted(
        occupation_results, key=lambda x: x["projection_score"], reverse=True
    )

    report_lines.append(f"\n【职业性别偏向分析 - 性别轴投影方法】")
    report_lines.append(f"\n最偏女性的职业 (Top 5):")
    for i, occ in enumerate(occupation_results_sorted_proj[:5], 1):
        report_lines.append(
            f"  {i}. {occ['occupation']:8s} | 投影分数: {occ['projection_score']:+.3f} "
            f"| 偏向分数: {occ['bias_score']:+.3f}"
        )

    report_lines.append(f"\n最偏男性的职业 (Top 5):")
    for i, occ in enumerate(occupation_results_sorted_proj[-5:][::-1], 1):
        report_lines.append(
            f"  {i}. {occ['occupation']:8s} | 投影分数: {occ['projection_score']:+.3f} "
            f"| 偏向分数: {occ['bias_score']:+.3f}"
        )

    # 计算家务分工词汇的性别偏向（类似职业词分析）
    domestic_work_results = []
    found_work_words = []
    found_family_words = []

    report_lines.append(f"\n【家务分工词汇性别偏向分析】")

    # 分析每个work词的性别偏向
    for word in DOMESTIC_WORK_WORDS["work"]:
        word_vec = get_word_embedding(model, word)
        if word_vec is not None:
            # 方法1：余弦相似度差值
            bias_score, male_sim, female_sim = compute_gender_bias(
                word_vec, male_vec, female_vec
            )

            # 方法2：性别轴投影
            projection_score, _ = compute_gender_bias_projection(
                word_vec, male_vec, female_vec
            )

            domestic_work_results.append(
                {
                    "word": word,
                    "word_type": "work",
                    "bias_score": float(bias_score),
                    "projection_score": float(projection_score),
                    "male_similarity": float(male_sim),
                    "female_similarity": float(female_sim),
                }
            )
            found_work_words.append(word)

    # 分析每个family词的性别偏向
    for word in DOMESTIC_WORK_WORDS["family"]:
        word_vec = get_word_embedding(model, word)
        if word_vec is not None:
            # 方法1：余弦相似度差值
            bias_score, male_sim, female_sim = compute_gender_bias(
                word_vec, male_vec, female_vec
            )

            # 方法2：性别轴投影
            projection_score, _ = compute_gender_bias_projection(
                word_vec, male_vec, female_vec
            )

            domestic_work_results.append(
                {
                    "word": word,
                    "word_type": "family",
                    "bias_score": float(bias_score),
                    "projection_score": float(projection_score),
                    "male_similarity": float(male_sim),
                    "female_similarity": float(female_sim),
                }
            )
            found_family_words.append(word)

    if not domestic_work_results:
        report_lines.append(f"  (未找到work/family词)")
    else:
        report_lines.append(
            f"找到work词: {len(found_work_words)}/{len(DOMESTIC_WORK_WORDS['work'])} 个"
        )
        report_lines.append(
            f"找到family词: {len(found_family_words)}/{len(DOMESTIC_WORK_WORDS['family'])} 个"
        )

        # 分别统计work和family词的性别偏向
        work_results = [r for r in domestic_work_results if r["word_type"] == "work"]
        family_results = [
            r for r in domestic_work_results if r["word_type"] == "family"
        ]

        if work_results:
            work_bias_scores = [r["bias_score"] for r in work_results]
            work_proj_scores = [r["projection_score"] for r in work_results]
            report_lines.append(f"\nWork词汇统计:")
            report_lines.append(f"  余弦相似度差值方法:")
            report_lines.append(f"    平均偏向: {np.mean(work_bias_scores):+.3f}")
            report_lines.append(f"    标准差: {np.std(work_bias_scores):.3f}")
            report_lines.append(f"  性别轴投影方法:")
            report_lines.append(f"    平均投影: {np.mean(work_proj_scores):+.3f}")
            report_lines.append(f"    标准差: {np.std(work_proj_scores):.3f}")

        if family_results:
            family_bias_scores = [r["bias_score"] for r in family_results]
            family_proj_scores = [r["projection_score"] for r in family_results]
            report_lines.append(f"\nFamily词汇统计:")
            report_lines.append(f"  余弦相似度差值方法:")
            report_lines.append(f"    平均偏向: {np.mean(family_bias_scores):+.3f}")
            report_lines.append(f"    标准差: {np.std(family_bias_scores):.3f}")
            report_lines.append(f"  性别轴投影方法:")
            report_lines.append(f"    平均投影: {np.mean(family_proj_scores):+.3f}")
            report_lines.append(f"    标准差: {np.std(family_proj_scores):.3f}")

        if work_results and family_results:
            bias_gap = np.mean(family_bias_scores) - np.mean(work_bias_scores)
            proj_gap = np.mean(family_proj_scores) - np.mean(work_proj_scores)
            report_lines.append(f"\nWork vs Family 性别差异:")
            report_lines.append(
                f"  余弦相似度差值: {bias_gap:+.3f} (正值表示family比work更偏女性)"
            )
            report_lines.append(
                f"  性别轴投影: {proj_gap:+.3f} (正值表示family比work更偏女性)"
            )

        # 展示最偏女性和最偏男性的词（按余弦相似度差值）
        sorted_results = sorted(
            domestic_work_results, key=lambda x: x["bias_score"], reverse=True
        )

        report_lines.append(f"\n最偏女性的work/family词 (Top 5):")
        for i, word_data in enumerate(sorted_results[:5], 1):
            report_lines.append(
                f"  {i}. [{word_data['word_type']:6s}] {word_data['word']:10s} | "
                f"偏向分数: {word_data['bias_score']:+.3f} | "
                f"投影分数: {word_data['projection_score']:+.3f}"
            )

        report_lines.append(f"\n最偏男性的work/family词 (Top 5):")
        for i, word_data in enumerate(sorted_results[-5:][::-1], 1):
            report_lines.append(
                f"  {i}. [{word_data['word_type']:6s}] {word_data['word']:10s} | "
                f"偏向分数: {word_data['bias_score']:+.3f} | "
                f"投影分数: {word_data['projection_score']:+.3f}"
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
        # 职业词：余弦相似度差值方法的统计
        "occupation_mean_bias": float(np.mean(bias_scores)),
        "occupation_std_bias": float(np.std(bias_scores)),
        "occupation_min_bias": float(np.min(bias_scores)),
        "occupation_max_bias": float(np.max(bias_scores)),
        "occupation_range_bias": float(np.max(bias_scores) - np.min(bias_scores)),
        # 职业词：性别轴投影方法的统计
        "occupation_mean_projection": float(np.mean(projection_scores)),
        "occupation_std_projection": float(np.std(projection_scores)),
        "occupation_min_projection": float(np.min(projection_scores)),
        "occupation_max_projection": float(np.max(projection_scores)),
        "occupation_range_projection": float(
            np.max(projection_scores) - np.min(projection_scores)
        ),
    }

    # 添加work/family词汇的统计指标
    if domestic_work_results:
        work_results = [r for r in domestic_work_results if r["word_type"] == "work"]
        family_results = [
            r for r in domestic_work_results if r["word_type"] == "family"
        ]

        stats["work_words_found"] = len(found_work_words)
        stats["family_words_found"] = len(found_family_words)

        if work_results:
            work_bias_scores = [r["bias_score"] for r in work_results]
            work_proj_scores = [r["projection_score"] for r in work_results]
            stats["work_mean_bias"] = float(np.mean(work_bias_scores))
            stats["work_std_bias"] = float(np.std(work_bias_scores))
            stats["work_mean_projection"] = float(np.mean(work_proj_scores))
            stats["work_std_projection"] = float(np.std(work_proj_scores))

        if family_results:
            family_bias_scores = [r["bias_score"] for r in family_results]
            family_proj_scores = [r["projection_score"] for r in family_results]
            stats["family_mean_bias"] = float(np.mean(family_bias_scores))
            stats["family_std_bias"] = float(np.std(family_bias_scores))
            stats["family_mean_projection"] = float(np.mean(family_proj_scores))
            stats["family_std_projection"] = float(np.std(family_proj_scores))

        # 计算work vs family的差异
        if work_results and family_results:
            stats["domain_bias_gap"] = float(
                np.mean(family_bias_scores) - np.mean(work_bias_scores)
            )
            stats["domain_projection_gap"] = float(
                np.mean(family_proj_scores) - np.mean(work_proj_scores)
            )

    report_lines.append(f"\n【统计指标汇总】")
    report_lines.append(f"\n职业词统计:")
    report_lines.append(f"  余弦相似度差值方法:")
    report_lines.append(f"    平均偏向: {stats['occupation_mean_bias']:+.3f}")
    report_lines.append(f"    标准差（隔离程度）: {stats['occupation_std_bias']:.3f}")
    report_lines.append(
        f"    偏向范围: [{stats['occupation_min_bias']:+.3f}, {stats['occupation_max_bias']:+.3f}]"
    )
    report_lines.append(f"  性别轴投影方法:")
    report_lines.append(f"    平均投影: {stats['occupation_mean_projection']:+.3f}")
    report_lines.append(f"    标准差: {stats['occupation_std_projection']:.3f}")
    report_lines.append(
        f"    投影范围: [{stats['occupation_min_projection']:+.3f}, {stats['occupation_max_projection']:+.3f}]"
    )

    if domestic_work_results:
        report_lines.append(f"\nWork/Family词统计:")
        if "work_mean_bias" in stats:
            report_lines.append(f"  Work词:")
            report_lines.append(f"    平均偏向: {stats['work_mean_bias']:+.3f}")
            report_lines.append(f"    平均投影: {stats['work_mean_projection']:+.3f}")
        if "family_mean_bias" in stats:
            report_lines.append(f"  Family词:")
            report_lines.append(f"    平均偏向: {stats['family_mean_bias']:+.3f}")
            report_lines.append(f"    平均投影: {stats['family_mean_projection']:+.3f}")
        if "domain_bias_gap" in stats:
            report_lines.append(f"  Domain差异:")
            report_lines.append(f"    偏向差距: {stats['domain_bias_gap']:+.3f}")
            report_lines.append(f"    投影差距: {stats['domain_projection_gap']:+.3f}")

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
        "work_words_found": found_work_words,
        "family_words_found": found_family_words,
        "report_lines": report_lines,  # 添加详细报告
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

    # 2.5. 保存work/family词汇性别偏向数据（类似职业数据）
    domestic_work_data = []
    for result in results:
        province = result["province"]
        if result.get("domestic_work_results"):
            for dw in result["domestic_work_results"]:
                domestic_work_data.append(
                    {
                        "province": province,
                        "word": dw["word"],
                        "word_type": dw["word_type"],
                        "bias_score": dw["bias_score"],
                        "projection_score": dw["projection_score"],
                        "male_similarity": dw["male_similarity"],
                        "female_similarity": dw["female_similarity"],
                    }
                )

    if domestic_work_data:
        domestic_work_df = pd.DataFrame(domestic_work_data)
        domestic_work_file = os.path.join(year_output_dir, f"domestic_work_bias.csv")
        domestic_work_df.to_csv(domestic_work_file, index=False, encoding="utf-8-sig")
        print(f"✓ Work/Family词汇性别偏向数据: {domestic_work_file}")

        # 保存宽格式数据（省份×词汇矩阵）
        # 2.5.1 余弦相似度差值方法的矩阵
        dw_pivot_df = domestic_work_df.pivot_table(
            values="bias_score", index="word", columns="province", aggfunc="mean"
        )
        dw_pivot_file = os.path.join(year_output_dir, f"domestic_work_bias_pivot.csv")
        dw_pivot_df.to_csv(dw_pivot_file, encoding="utf-8-sig")
        print(f"✓ Work/Family词×省份矩阵（余弦相似度差值）: {dw_pivot_file}")

        # 2.5.2 性别轴投影方法的矩阵
        dw_pivot_proj_df = domestic_work_df.pivot_table(
            values="projection_score", index="word", columns="province", aggfunc="mean"
        )
        dw_pivot_proj_file = os.path.join(
            year_output_dir, f"domestic_work_projection_pivot.csv"
        )
        dw_pivot_proj_df.to_csv(dw_pivot_proj_file, encoding="utf-8-sig")
        print(f"✓ Work/Family词×省份矩阵（性别轴投影）: {dw_pivot_proj_file}")

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

    # 5. 生成详细分析报告
    report_file = os.path.join(year_output_dir, f"analysis_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"{'='*80}\n")
        f.write(f"性别-职业Embedding分析报告 ({year}年)\n")
        f.write(f"{'='*80}\n\n")

        f.write(f"分析省份数: {len(results)}\n")
        f.write(f"分析职业数: {len(ALL_OCCUPATIONS)}\n\n")

        # 写入每个省份的详细分析报告
        f.write(f"\n{'#'*80}\n")
        f.write(f"# 各省份详细分析\n")
        f.write(f"{'#'*80}\n")

        for result in results:
            if "report_lines" in result:
                f.write("\n")
                for line in result["report_lines"]:
                    f.write(f"{line}\n")

        # 分隔符
        f.write(f"\n\n{'#'*80}\n")
        f.write(f"# 跨省份汇总统计\n")
        f.write(f"{'#'*80}\n\n")

        f.write(f"{'='*60}\n")
        f.write(f"各省份职业性别隔离指数排名（余弦相似度差值方法，按标准差）:\n")
        f.write(f"{'='*60}\n")
        stats_sorted = sorted(
            province_stats, key=lambda x: x["occupation_std_bias"], reverse=True
        )
        for i, stat in enumerate(stats_sorted, 1):
            f.write(
                f"{i:2d}. {stat['province']:10s} | "
                f"隔离指数: {stat['occupation_std_bias']:.3f} | "
                f"平均偏向: {stat['occupation_mean_bias']:+.3f}\n"
            )

        f.write(f"\n{'='*60}\n")
        f.write(f"各省份职业性别隔离指数排名（性别轴投影方法，按标准差）:\n")
        f.write(f"{'='*60}\n")
        stats_sorted_proj = sorted(
            province_stats,
            key=lambda x: x.get("occupation_std_projection", 0),
            reverse=True,
        )
        for i, stat in enumerate(stats_sorted_proj, 1):
            f.write(
                f"{i:2d}. {stat['province']:10s} | "
                f"隔离指数: {stat.get('occupation_std_projection', 0):.3f} | "
                f"平均投影: {stat.get('occupation_mean_projection', 0):+.3f}\n"
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

        # 添加work/family词汇性别偏向分析
        if domestic_work_data:
            f.write(f"\n{'='*60}\n")
            f.write(f"Work/Family词汇性别偏向分析:\n")
            f.write(f"{'='*60}\n")

            domestic_work_df = pd.DataFrame(domestic_work_data)

            # 各省份work vs family平均偏向差异排名
            f.write(f"\n各省份Work vs Family性别差异排名（余弦相似度差值方法）:\n")
            f.write(f"(正值表示family比work更偏女性)\n")
            province_gaps = []
            for stat in province_stats:
                if "domain_bias_gap" in stat:
                    province_gaps.append(
                        {
                            "province": stat["province"],
                            "gap": stat["domain_bias_gap"],
                        }
                    )
            if province_gaps:
                province_gaps_df = pd.DataFrame(province_gaps)
                province_gaps_df = province_gaps_df.sort_values("gap", ascending=False)
                for i, row in enumerate(province_gaps_df.itertuples(), 1):
                    f.write(f"  {i:2d}. {row.province:10s} | 差异: {row.gap:+.3f}\n")

            # 各省份work vs family平均投影差异排名
            f.write(f"\n各省份Work vs Family性别差异排名（性别轴投影方法）:\n")
            f.write(f"(正值表示family比work更偏女性)\n")
            province_proj_gaps = []
            for stat in province_stats:
                if "domain_projection_gap" in stat:
                    province_proj_gaps.append(
                        {
                            "province": stat["province"],
                            "gap": stat["domain_projection_gap"],
                        }
                    )
            if province_proj_gaps:
                province_proj_gaps_df = pd.DataFrame(province_proj_gaps)
                province_proj_gaps_df = province_proj_gaps_df.sort_values(
                    "gap", ascending=False
                )
                for i, row in enumerate(province_proj_gaps_df.itertuples(), 1):
                    f.write(f"  {i:2d}. {row.province:10s} | 差异: {row.gap:+.3f}\n")

            # Work/Family词汇一致性分析
            f.write(f"\n{'='*60}\n")
            f.write(f"Work/Family词汇性别偏向一致性分析:\n")
            f.write(f"{'='*60}\n")

            # 计算每个词汇在各省份的平均偏向
            word_avg = (
                domestic_work_df.groupby("word")["bias_score"]
                .agg(["mean", "std"])
                .sort_values("mean", ascending=False)
            )

            f.write(f"\n最偏女性的work/family词（跨省份平均）:\n")
            for i, (word, row) in enumerate(word_avg.head(10).iterrows(), 1):
                word_type = domestic_work_df[domestic_work_df["word"] == word][
                    "word_type"
                ].iloc[0]
                f.write(
                    f"  {i:2d}. [{word_type:6s}] {word:15s} | 平均: {row['mean']:+.3f} | 标准差: {row['std']:.3f}\n"
                )

            f.write(f"\n最偏男性的work/family词（跨省份平均）:\n")
            for i, (word, row) in enumerate(word_avg.tail(10).iloc[::-1].iterrows(), 1):
                word_type = domestic_work_df[domestic_work_df["word"] == word][
                    "word_type"
                ].iloc[0]
                f.write(
                    f"  {i:2d}. [{word_type:6s}] {word:15s} | 平均: {row['mean']:+.3f} | 标准差: {row['std']:.3f}\n"
                )

            # 分别统计work和family词汇的跨省份平均
            work_df = domestic_work_df[domestic_work_df["word_type"] == "work"]
            family_df = domestic_work_df[domestic_work_df["word_type"] == "family"]

            if not work_df.empty and not family_df.empty:
                work_mean = work_df["bias_score"].mean()
                family_mean = family_df["bias_score"].mean()
                work_proj_mean = work_df["projection_score"].mean()
                family_proj_mean = family_df["projection_score"].mean()

                f.write(f"\n跨省份平均性别偏向:\n")
                f.write(f"  余弦相似度差值方法:\n")
                f.write(f"    Work词汇: {work_mean:+.3f}\n")
                f.write(f"    Family词汇: {family_mean:+.3f}\n")
                f.write(f"    差异: {family_mean - work_mean:+.3f}\n")
                f.write(f"  性别轴投影方法:\n")
                f.write(f"    Work词汇: {work_proj_mean:+.3f}\n")
                f.write(f"    Family词汇: {family_proj_mean:+.3f}\n")
                f.write(f"    差异: {family_proj_mean - work_proj_mean:+.3f}\n")

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

    # 获取要分析的省份列表
    if province:
        provinces_to_analyze = [province]
        print(f"🎯 分析指定省份: {province}\n")
    else:
        provinces_to_analyze = get_available_provinces(year)
        if not provinces_to_analyze:
            print(f"❌ 未找到 {year} 年的模型文件")
            return
        print(f"📂 找到 {len(provinces_to_analyze)} 个省份，将逐个分析\n")

    # 逐个加载和分析省份模型（节省内存）
    results = []
    province_stats = []

    for idx, province_name in enumerate(provinces_to_analyze, 1):
        print(f"\n{'='*60}")
        print(f"处理进度: [{idx}/{len(provinces_to_analyze)}] {province_name}")
        print(f"{'='*60}")

        # 加载单个模型
        model = load_single_model(year, province_name)
        if model is None:
            print(f"  ⚠️  跳过省份: {province_name}")
            continue

        # 分析单个模型
        result = analyze_model(province_name, model)
        if result:
            results.append(result)
            province_stats.append(result["stats"])

        # 释放模型内存
        del model
        import gc

        gc.collect()

    if not results:
        print("❌ 没有生成任何结果")
        return

    # 保存结果
    save_results(results, province_stats, year)

    print(f"\n{'='*60}")
    print(f"🎉 {year} 年embedding分析完成！共分析 {len(results)} 个省份")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    fire.Fire(main)
