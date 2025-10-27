"""
分析男性和女性讨论内容与官号发布新闻的相关性

功能：
1. 根据官号ID列表，提取官号发布的微博内容
2. 分别提取男性和女性的讨论内容
3. 使用embedding计算内容相似性
4. 比较不同性别与官号内容的接近程度

输入数据：cleaned_weibo_cov/{year}/ 下的parquet文件
"""

import os
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import jieba
import fire
from sklearn.preprocessing import normalize
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")

DATA_DIR = "cleaned_weibo_cov"
OUTPUT_DIR = "content_similarity_analysis"

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 官方账号ID列表（待填充）
OFFICIAL_ACCOUNT_IDS = set()

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
        "她",
        "他",
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


def load_data_for_year(year):
    """加载指定年份的所有数据"""
    year_dir = os.path.join(DATA_DIR, str(year))
    if not os.path.exists(year_dir):
        print(f"未找到 {year} 年的数据目录: {year_dir}")
        return None

    import glob

    pattern = os.path.join(year_dir, "*.parquet")
    parquet_files = sorted(glob.glob(pattern))

    if not parquet_files:
        print(f"未找到 {year} 年的parquet文件")
        return None

    print(f"找到 {len(parquet_files)} 个文件，开始加载...")

    all_data = []
    for file_path in parquet_files:
        try:
            df = pd.read_parquet(file_path)
            # 从文件名提取日期
            filename = os.path.basename(file_path)
            date_str = filename.replace(".parquet", "")
            df["date"] = date_str  # 添加日期列
            all_data.append(df)
        except Exception as e:
            print(f"读取文件 {file_path} 失败: {e}")
            continue

    if not all_data:
        print(f"未能加载 {year} 年的任何数据")
        return None

    result = pd.concat(all_data, ignore_index=True)
    print(f"共加载 {len(result)} 条数据")
    return result


def extract_content_by_group(data):
    """根据性别分组提取内容"""

    # 确定性别字段
    gender_col = (
        "demographic_gender" if "demographic_gender" in data.columns else "gender"
    )

    if gender_col not in data.columns:
        print("警告: 数据中没有性别字段")
        return None

    # 过滤空值和空内容
    data = data.dropna(subset=[gender_col, "weibo_content"])

    # 提取官方账号内容
    official_texts = []
    if OFFICIAL_ACCOUNT_IDS:
        official_data = data[data["user_id"].isin(OFFICIAL_ACCOUNT_IDS)]
        official_texts = [
            preprocess_text(row["weibo_content"]) for _, row in official_data.iterrows()
        ]
        official_texts = [t for t in official_texts if len(t) > 3]
        print(f"官方账号内容: {len(official_texts)} 条")

    # 提取男性内容
    male_data = data[
        (data[gender_col].isin(["男", "male", "M", "1"]))
        & (~data["user_id"].isin(OFFICIAL_ACCOUNT_IDS))
    ]
    male_texts = [
        preprocess_text(row["weibo_content"]) for _, row in male_data.iterrows()
    ]
    male_texts = [t for t in male_texts if len(t) > 3]
    print(f"男性用户内容: {len(male_texts)} 条")

    # 提取女性内容
    female_data = data[
        (data[gender_col].isin(["女", "female", "F", "0"]))
        & (~data["user_id"].isin(OFFICIAL_ACCOUNT_IDS))
    ]
    female_texts = [
        preprocess_text(row["weibo_content"]) for _, row in female_data.iterrows()
    ]
    female_texts = [t for t in female_texts if len(t) > 3]
    print(f"女性用户内容: {len(female_texts)} 条")

    return {
        "official": official_texts,
        "male": male_texts,
        "female": female_texts,
    }


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


def get_text_embedding(model, texts, mode="mean"):
    """获取一组文本的平均或最大向量"""
    vectors = []
    for text in texts:
        text_vecs = []
        for word in text:
            try:
                vec = model.wv[word]
                text_vecs.append(vec)
            except KeyError:
                continue

        if text_vecs:
            if mode == "mean":
                text_vec = np.mean(text_vecs, axis=0)
            elif mode == "max":
                text_vec = np.max(text_vecs, axis=0)
            else:
                text_vec = np.mean(text_vecs, axis=0)
            vectors.append(text_vec)

    if not vectors:
        return None

    # 返回整个集合的平均向量
    avg_vec = np.mean(vectors, axis=0)
    return normalize([avg_vec])[0]


def cosine_similarity(vec1, vec2):
    """计算余弦相似度"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def analyze_content_similarity(year):
    """分析内容相似性"""

    print(f"\n开始分析 {year} 年内容相似性...")

    # 加载数据
    data = load_data_for_year(year)
    if data is None:
        return

    # 检查官方账号ID
    if not OFFICIAL_ACCOUNT_IDS:
        print("警告: 官方账号ID列表为空")
        print("请使用命令添加官方账号ID:")
        print("  python content_similarity_analyzer.py add_account --account_id 123456")
        print("或直接在脚本中修改 OFFICIAL_ACCOUNT_IDS 变量")
        return

    print(f"\n使用官方账号ID: {OFFICIAL_ACCOUNT_IDS}")

    # 提取内容
    content_groups = extract_content_by_group(data)
    if content_groups is None:
        print("无法提取内容")
        return

    # 检查数据量
    if len(content_groups["official"]) < 100:
        print("官方账号内容不足100条，无法进行有意义的分析")
        return

    if len(content_groups["male"]) < 100:
        print("男性内容不足100条，无法进行有意义的分析")
        return

    if len(content_groups["female"]) < 100:
        print("女性内容不足100条，无法进行有意义的分析")
        return

    # 训练模型
    print(f"\n训练Word2Vec模型...")

    # 使用所有文本训练
    all_texts = (
        content_groups["official"] + content_groups["male"] + content_groups["female"]
    )
    model = train_word2vec(all_texts)

    if model is None:
        print("训练模型失败")
        return

    print("模型训练完成")

    # 获取各组内容的embedding
    print(f"\n计算内容embedding...")

    official_vec = get_text_embedding(model, content_groups["official"])
    male_vec = get_text_embedding(model, content_groups["male"])
    female_vec = get_text_embedding(model, content_groups["female"])

    if official_vec is None or male_vec is None or female_vec is None:
        print("计算embedding失败")
        return

    # 计算相似度
    male_similarity = cosine_similarity(male_vec, official_vec)
    female_similarity = cosine_similarity(female_vec, official_vec)

    print(f"\n相似性分析结果:")
    print(f"  男性与官号内容相似度: {male_similarity:.4f}")
    print(f"  女性与官号内容相似度: {female_similarity:.4f}")
    print(f"  性别差异 (男性-女性): {male_similarity - female_similarity:.4f}")

    # 保存结果
    result = {
        "year": year,
        "official_account_count": len(content_groups["official"]),
        "male_content_count": len(content_groups["male"]),
        "female_content_count": len(content_groups["female"]),
        "male_similarity": float(male_similarity),
        "female_similarity": float(female_similarity),
        "similarity_diff": float(male_similarity - female_similarity),
        "official_vec": official_vec.tolist(),
        "male_vec": male_vec.tolist(),
        "female_vec": female_vec.tolist(),
    }

    # 保存为DataFrame
    result_df = pd.DataFrame([result])
    output_file = os.path.join(OUTPUT_DIR, f"similarity_analysis_{year}.parquet")
    result_df.to_parquet(output_file, engine="fastparquet", index=False)
    print(f"\n分析结果已保存到: {output_file}")

    # 保存模型
    model_path = os.path.join(OUTPUT_DIR, f"model_{year}.model")
    model.save(model_path)
    print(f"模型已保存到: {model_path}")

    print(f"\n{year} 年内容相似性分析完成！")


def add_account(account_id: str):
    """
    添加官方账号ID到列表中

    Args:
        account_id: 官方账号ID
    """
    global OFFICIAL_ACCOUNT_IDS

    # 从配置文件读取已存在的ID
    config_file = os.path.join("configs", "official_account_ids.json")
    if os.path.exists(config_file):
        import json

        with open(config_file, "r") as f:
            OFFICIAL_ACCOUNT_IDS = set(json.load(f))

    OFFICIAL_ACCOUNT_IDS.add(account_id)

    # 保存到配置文件
    os.makedirs("configs", exist_ok=True)
    with open(config_file, "w") as f:
        json.dump(list(OFFICIAL_ACCOUNT_IDS), f, indent=2)

    print(f"已添加官方账号ID: {account_id}")
    print(f"当前官方账号ID列表: {list(OFFICIAL_ACCOUNT_IDS)}")


def load_accounts_from_config():
    """从配置文件加载官方账号ID"""
    global OFFICIAL_ACCOUNT_IDS

    config_file = os.path.join("configs", "official_account_ids.json")
    if os.path.exists(config_file):
        import json

        with open(config_file, "r") as f:
            OFFICIAL_ACCOUNT_IDS = set(json.load(f))
            print(f"从配置文件加载了 {len(OFFICIAL_ACCOUNT_IDS)} 个官方账号ID")
    else:
        print("未找到官方账号配置文件，使用脚本中的默认列表")


def main(year: int):
    """
    运行内容相似性分析

    Args:
        year: 年份
    """
    # 先尝试加载配置文件中的账号ID
    load_accounts_from_config()

    analyze_content_similarity(year)


if __name__ == "__main__":
    fire.Fire({"analyze": main, "add_account": add_account})
