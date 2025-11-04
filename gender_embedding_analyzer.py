"""
性别和职业词的embedding分析

功能：
1. 按省份分组数据训练Word2Vec模型
2. 计算职业词与性别词的关联度（分别计算与男性词、女性词的相似度）
3. 比较不同省份模型的差异

输入数据：cleaned_weibo_cov/{year}/ 下的parquet文件
"""

import os
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from collections import defaultdict
import jieba
import fire
from sklearn.preprocessing import normalize
import warnings
import json

warnings.filterwarnings("ignore")

DATA_DIR = "cleaned_weibo_cov"
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

# 通用停用词
STOPWORDS = set(
    [
        "的",
        "是",
        "了",
        "在",
        "有",
        "和",
        "就",
        "不",
        "人",
        "都",
        "一",
        "一个",
        "上",
        "也",
        "很",
        "到",
        "说",
        "要",
        "去",
        "你",
        "会",
        "着",
        "没有",
        "看",
        "好",
        "自己",
        "这",
        "那",
        "我",
        "他",
        "她",
        "我们",
        "你们",
        "他们",
        "她们",
        "什么",
        "怎么",
        "这个",
        "那个",
    ]
)


def preprocess_text(text):
    """预处理文本，分词并过滤停用词"""
    if pd.isna(text) or text == "":
        return []

    text = str(text)
    words = jieba.cut(text)
    words = [
        w.strip()
        for w in words
        if w.strip() and w not in STOPWORDS and len(w.strip()) > 1
    ]
    return words


def load_data_by_province(year):
    """按省份加载数据（内存优化版本）"""
    year_dir = os.path.join(DATA_DIR, str(year))
    if not os.path.exists(year_dir):
        print(f"❌ 未找到 {year} 年的数据目录")
        return None

    import glob

    pattern = os.path.join(year_dir, "*.parquet")
    parquet_files = sorted(glob.glob(pattern))

    if not parquet_files:
        print(f"❌ 未找到 {year} 年的数据文件")
        return None

    print(f"📂 找到 {len(parquet_files)} 个文件，开始加载...")

    # 只加载需要的列，减少内存占用
    required_columns = ["weibo_content"]
    province_col = None

    # 先检查第一个文件确定省份字段名（只读取一行，减少内存占用）
    province_col = "province"
    required_columns.append("province")

    if province_col is None:
        print(f"❌ 无法确定省份字段")
        return None

    data_by_province = defaultdict(list)

    for file_idx, file_path in enumerate(parquet_files):
        try:
            # 只读取需要的列
            df = pd.read_parquet(file_path, columns=required_columns)

            # 过滤掉空值
            df = df.dropna(subset=[province_col, "weibo_content"])

            # 按省份分组，使用字典直接聚合而不是append
            for province in df[province_col].unique():
                province_data = df[df[province_col] == province].copy()
                data_by_province[province].append(province_data)

            # 及时释放内存
            del df

            if (file_idx + 1) % 10 == 0:
                print(f"  已处理 {file_idx + 1}/{len(parquet_files)} 个文件...")

        except Exception as e:
            print(f"❌ 读取文件 {file_path} 失败: {e}")
            continue

    # 合并每个省份的数据（使用concat但及时释放）
    print(f"\n📊 按省份分组，共 {len(data_by_province)} 个省份")

    result = {}
    for province, data_list in data_by_province.items():
        # 合并数据
        combined_data = pd.concat(data_list, ignore_index=True)
        # 立即释放原列表内存
        del data_list

        if len(combined_data) > 1000:  # 至少1000条数据
            result[province] = combined_data
            print(f"  ✓ {province}: {len(combined_data):,} 条数据")
        else:
            print(f"  ✗ {province}: {len(combined_data):,} 条数据 (跳过，数据量不足)")
            del combined_data

    return result


def train_word2vec(texts, vector_size=300, window=5, min_count=20, workers=None):
    """
    训练Word2Vec模型（内存优化版本）

    参数调整说明：
    - vector_size: 300（更大的向量维度，更好的语义表达）
    - window: 5（上下文窗口）
    - min_count: 20（词频阈值，根据数据量调整）
    - workers: 线程数，None则自动设置为CPU核心数-1
    """
    if not texts or len(texts) < 100:
        return None

    # 自动设置workers，避免超过CPU核心数
    if workers is None:
        import multiprocessing

        workers = max(1, multiprocessing.cpu_count() - 1)

    # 限制workers数量，避免内存过度占用
    workers = min(workers, 8)

    model = Word2Vec(
        sentences=texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,  # 多线程，但限制数量
        epochs=10,
        sg=1,  # Skip-gram（对中小规模数据更好）
        negative=10,  # 负采样
        seed=42,  # 可重复性
        max_vocab_size=None,  # 不限制词汇表大小，但可以通过min_count控制
    )

    return model


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


def analyze_province_embedding(data_by_province, year):
    """分析每个省份的embedding"""
    results = []
    province_stats = []

    for province, data in data_by_province.items():
        print(f"\n{'='*60}")
        print(f"🔍 处理省份: {province}")
        print(f"{'='*60}")

        # 预处理文本（内存优化：使用itertuples而不是iterrows，分批处理）
        # 先保存数据条数，因为后面会删除DataFrame
        data_count = len(data)

        texts = []
        # 使用itertuples比iterrows快得多且内存占用更少
        for row in data.itertuples():
            words = preprocess_text(row.weibo_content)
            if len(words) > 3:  # 至少3个词
                texts.append(words)

        # 处理完文本后立即释放DataFrame内存
        del data

        if len(texts) < 100:
            print(f"  ❌ 文本量不足 ({len(texts)} 条)，跳过")
            del texts
            continue

        print(f"  📝 有效文本: {len(texts):,} 条")

        # 训练模型
        print(f"  🔧 训练Word2Vec模型...")
        model = train_word2vec(texts)
        if model is None:
            print(f"  ❌ 训练模型失败")
            continue

        vocab_size = len(model.wv)
        print(f"  ✓ 模型训练完成，词汇表大小: {vocab_size:,}")

        # 训练完成后立即释放texts列表（可能占用大量内存）
        text_count = len(texts)
        del texts
        import gc

        gc.collect()  # 强制垃圾回收

        # 计算性别词向量
        male_vec, male_found = get_word_set_embedding(model, GENDER_WORDS["male"])
        female_vec, female_found = get_word_set_embedding(model, GENDER_WORDS["female"])

        if male_vec is None or female_vec is None:
            print(f"  ❌ 性别词向量计算失败")
            continue

        print(f"  ✓ 找到男性词: {len(male_found)}/{len(GENDER_WORDS['male'])} 个")
        print(
            f"    {', '.join(male_found[:10])}{'...' if len(male_found) > 10 else ''}"
        )
        print(f"  ✓ 找到女性词: {len(female_found)}/{len(GENDER_WORDS['female'])} 个")
        print(
            f"    {', '.join(female_found[:10])}{'...' if len(female_found) > 10 else ''}"
        )

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
            continue

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
            "data_count": data_count,
            "text_count": text_count,
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
        province_stats.append(stats)

        print(f"\n  📈 统计指标:")
        print(f"    平均偏向: {stats['mean_bias']:+.3f}")
        print(f"    标准差（隔离程度）: {stats['std_bias']:.3f}")
        print(f"    偏向范围: [{stats['min_bias']:+.3f}, {stats['max_bias']:+.3f}]")

        # 保存详细结果（先转换向量为列表，避免后续内存占用）
        result = {
            "province": province,
            "stats": stats,
            "male_vec": male_vec.tolist(),  # 转换为列表后，原始numpy数组可以释放
            "female_vec": female_vec.tolist(),
            "male_words_found": male_found,
            "female_words_found": female_found,
            "occupations_found": found_occupations,
            "occupation_results": occupation_results,
        }
        results.append(result)

        # 保存结果后立即释放向量（已经转换为列表，原始numpy数组不再需要）
        del male_vec
        del female_vec

        # 保存模型
        model_path = os.path.join(OUTPUT_DIR, f"model_{year}_{province}.model")
        model.save(model_path)
        print(f"  💾 模型已保存: {model_path}")

        # 保存模型后释放模型（释放内存）
        del model
        gc.collect()  # 再次垃圾回收

    return results, province_stats


def save_results(results, province_stats, year):
    """保存分析结果"""
    if not results:
        print("❌ 没有生成任何结果")
        return

    print(f"\n{'='*60}")
    print(f"💾 保存结果...")
    print(f"{'='*60}")

    # 1. 保存省份统计信息
    stats_df = pd.DataFrame(province_stats)
    stats_file = os.path.join(OUTPUT_DIR, f"province_stats_{year}.csv")
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
    occupation_file = os.path.join(OUTPUT_DIR, f"occupation_bias_{year}.csv")
    occupation_df.to_csv(occupation_file, index=False, encoding="utf-8-sig")
    print(f"✓ 职业性别偏向数据: {occupation_file}")

    # 3. 保存宽格式数据（省份×职业矩阵）
    pivot_df = occupation_df.pivot_table(
        values="bias_score", index="occupation", columns="province", aggfunc="mean"
    )
    pivot_file = os.path.join(OUTPUT_DIR, f"occupation_bias_pivot_{year}.csv")
    pivot_df.to_csv(pivot_file, encoding="utf-8-sig")
    print(f"✓ 职业×省份矩阵: {pivot_file}")

    # 4. 保存详细向量数据（JSON格式，便于后续分析）
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

    detailed_file = os.path.join(OUTPUT_DIR, f"detailed_vectors_{year}.json")
    with open(detailed_file, "w", encoding="utf-8") as f:
        json.dump(detailed_data, f, ensure_ascii=False, indent=2)
    print(f"✓ 详细向量数据: {detailed_file}")

    # 5. 生成简要分析报告
    report_file = os.path.join(OUTPUT_DIR, f"analysis_report_{year}.txt")
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

    print(f"\n✅ 所有结果已保存到 {OUTPUT_DIR}/ 目录")


def main(year: int, province: str = None):
    """
    运行embedding分析

    Args:
        year: 年份
        province: 指定省份（可选），如果不指定则处理所有省份
    """
    print(f"\n{'='*60}")
    print(f"🚀 开始分析 {year} 年数据的性别-职业Embedding")
    print(f"{'='*60}\n")

    # 加载数据
    data_by_province = load_data_by_province(year)
    if not data_by_province:
        print("❌ 无法加载数据")
        return

    # 如果指定了省份，只处理该省份
    if province:
        if province not in data_by_province:
            print(f"❌ 未找到省份: {province}")
            print(f"可用省份: {', '.join(data_by_province.keys())}")
            return
        data_by_province = {province: data_by_province[province]}
        print(f"🎯 只处理省份: {province}\n")

    # 分析embedding
    results, province_stats = analyze_province_embedding(data_by_province, year)

    # 保存结果
    save_results(results, province_stats, year)

    print(f"\n{'='*60}")
    print(f"🎉 {year} 年embedding分析完成！")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    fire.Fire(main)
