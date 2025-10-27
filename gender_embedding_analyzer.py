"""
性别和职业词的embedding分析

功能：
1. 按省份分组数据训练Word2Vec模型
2. 计算性别词表与职业词表的夹角
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

warnings.filterwarnings("ignore")

DATA_DIR = "cleaned_weibo_cov"
OUTPUT_DIR = "embedding_analysis"

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 性别词表
GENDER_WORDS = {
    "male": [
        "男",
        "男人",
        "男性",
        "帅哥",
        "先生",
        "爸爸",
        "父亲",
        "老公",
        "兄弟",
        "男朋友",
        "儿子",
        "爷们",
        "小伙子",
    ],
    "female": [
        "女",
        "女人",
        "女性",
        "美女",
        "女士",
        "妈妈",
        "母亲",
        "老婆",
        "姐妹",
        "女朋友",
        "女儿",
        "闺女",
        "小姑娘",
    ],
}

# 职业词表
OCCUPATION_WORDS = [
    "医生",
    "护士",
    "教师",
    "工程师",
    "程序员",
    "设计师",
    "律师",
    "会计",
    "销售",
    "经理",
    "老板",
    "服务员",
    "厨师",
    "警察",
    "消防员",
    "军人",
    "科学家",
    "研究员",
    "记者",
    "编辑",
    "作家",
    "演员",
    "歌手",
    "舞蹈",
    "主播",
    "网红",
]

# 通用停用词（可以扩展）
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
    """按省份加载数据"""
    year_dir = os.path.join(DATA_DIR, str(year))
    if not os.path.exists(year_dir):
        print(f"未找到 {year} 年的数据目录")
        return None

    import glob

    pattern = os.path.join(year_dir, "*.parquet")
    parquet_files = sorted(glob.glob(pattern))

    if not parquet_files:
        print(f"未找到 {year} 年的数据文件")
        return None

    print(f"找到 {len(parquet_files)} 个文件，开始加载...")

    data_by_province = defaultdict(list)

    for file_path in parquet_files:
        try:
            df = pd.read_parquet(file_path)

            # 确定省份字段
            province_col = (
                "demographic_province"
                if "demographic_province" in df.columns
                else "province"
            )

            if province_col not in df.columns:
                print(f"警告: 文件 {file_path} 没有省份信息，跳过")
                continue

            # 过滤掉空值
            df = df.dropna(subset=[province_col])

            # 按省份分组
            for province in df[province_col].unique():
                province_data = df[df[province_col] == province]
                data_by_province[province].append(province_data)

        except Exception as e:
            print(f"读取文件 {file_path} 失败: {e}")
            continue

    # 合并每个省份的数据
    print(f"按省份分组，共 {len(data_by_province)} 个省份")

    result = {}
    for province, data_list in data_by_province.items():
        combined_data = pd.concat(data_list, ignore_index=True)

        # 过滤掉内容为空的行
        combined_data = combined_data.dropna(subset=["weibo_content"])

        if len(combined_data) > 1000:  # 至少1000条数据
            result[province] = combined_data
            print(f"  {province}: {len(combined_data)} 条数据")
        else:
            print(f"  {province}: {len(combined_data)} 条数据 (跳过，数据量不足)")

    return result


def train_word2vec(texts, vector_size=100, window=5, min_count=5):
    """训练Word2Vec模型"""
    if not texts or len(texts) < 100:
        return None

    model = Word2Vec(
        sentences=texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        epochs=10,
    )

    return model


def get_word_embedding(model, word):
    """获取词向量"""
    try:
        return model.wv[word]
    except KeyError:
        return None


def get_word_set_embedding(model, words):
    """获取一组词的平均向量"""
    vectors = []
    for word in words:
        vec = get_word_embedding(model, word)
        if vec is not None:
            vectors.append(vec)

    if not vectors:
        return None

    # 计算平均向量
    avg_vec = np.mean(vectors, axis=0)
    return normalize([avg_vec])[0]


def cosine_similarity(vec1, vec2):
    """计算余弦相似度"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def analyze_province_embedding(data_by_province, year):
    """分析每个省份的embedding"""
    results = []

    for province, data in data_by_province.items():
        print(f"\n处理省份: {province}")

        # 预处理文本
        texts = []
        for _, row in data.iterrows():
            words = preprocess_text(row["weibo_content"])
            if len(words) > 3:  # 至少3个词
                texts.append(words)

        if len(texts) < 100:
            print(f"  文本量不足 ({len(texts)} 条)，跳过")
            continue

        print(f"  有效文本: {len(texts)} 条")

        # 训练模型
        model = train_word2vec(texts)
        if model is None:
            print(f"  训练模型失败")
            continue

        # 计算性别词向量
        male_vec = get_word_set_embedding(model, GENDER_WORDS["male"])
        female_vec = get_word_set_embedding(model, GENDER_WORDS["female"])

        if male_vec is None or female_vec is None:
            print(f"  性别词向量计算失败")
            continue

        # 计算性别向量差值
        gender_diff = male_vec - female_vec

        # 计算每个职业词与性别差值的余弦相似度
        occupation_similarities = {}

        for occupation in OCCUPATION_WORDS:
            occ_vec = get_word_embedding(model, occupation)
            if occ_vec is not None:
                # 计算与性别差值的夹角
                similarity = cosine_similarity(occ_vec, gender_diff)
                occupation_similarities[occupation] = float(similarity)

        # 保存结果
        result = {
            "province": province,
            "data_count": len(data),
            "text_count": len(texts),
            "male_vec": male_vec.tolist(),
            "female_vec": female_vec.tolist(),
            "gender_diff_vec": gender_diff.tolist(),
            "occupation_similarities": occupation_similarities,
        }

        results.append(result)

        # 打印前5个最"男性化"的职业
        sorted_occ = sorted(
            occupation_similarities.items(), key=lambda x: x[1], reverse=True
        )
        print(f"  最'男性化'的职业 (前5):")
        for occ, sim in sorted_occ[:5]:
            print(f"    {occ}: {sim:.3f}")

        # 保存模型
        model_path = os.path.join(OUTPUT_DIR, f"model_{year}_{province}.model")
        model.save(model_path)
        print(f"  模型已保存: {model_path}")

    return results


def main(year: int, province: str = None):
    """
    运行embedding分析

    Args:
        year: 年份
        province: 指定省份（可选），如果不指定则处理所有省份
    """
    print(f"开始分析 {year} 年数据的embedding...")

    # 加载数据
    data_by_province = load_data_by_province(year)
    if not data_by_province:
        print("无法加载数据")
        return

    # 如果指定了省份，只处理该省份
    if province:
        if province not in data_by_province:
            print(f"未找到省份: {province}")
            return
        data_by_province = {province: data_by_province[province]}
        print(f"只处理省份: {province}")

    # 分析embedding
    results = analyze_province_embedding(data_by_province, year)

    if not results:
        print("没有生成任何结果")
        return

    # 保存结果
    results_df = pd.DataFrame(
        [
            {
                "province": r["province"],
                "data_count": r["data_count"],
                "text_count": r["text_count"],
            }
            | r["occupation_similarities"]
            for r in results
        ]
    )

    output_file = os.path.join(OUTPUT_DIR, f"embedding_analysis_{year}.parquet")
    results_df.to_parquet(output_file, engine="fastparquet", index=False)
    print(f"\n分析结果已保存到: {output_file}")

    # 保存详细向量数据
    detailed_file = os.path.join(OUTPUT_DIR, f"detailed_vectors_{year}.parquet")
    detailed_df = pd.DataFrame(
        [
            {
                "province": r["province"],
                "data_count": r["data_count"],
                "male_vec": r["male_vec"],
                "female_vec": r["female_vec"],
                "gender_diff_vec": r["gender_diff_vec"],
            }
            for r in results
        ]
    )
    detailed_df.to_parquet(detailed_file, engine="fastparquet", index=False)
    print(f"详细向量数据已保存到: {detailed_file}")

    print(f"\n{year} 年embedding分析完成！")


if __name__ == "__main__":
    fire.Fire(main)
