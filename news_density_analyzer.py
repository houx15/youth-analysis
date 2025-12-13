"""
新闻密度分析脚本

从词汇角度分析男性和女性讨论内容与新闻官方媒体的重叠情况

功能：
1. 从官方媒体账号发布的内容中提取核心词汇（分词、提取动名词、TF-IDF，保留5000词）
2. 分析这些词在男性和女性发表内容中的出现频次，计算"news density"
3. 比较不同性别的平均news density和分布差异（Post级别、User级别、省份级别）
"""

import os
import pandas as pd
import numpy as np
import jieba.posseg as pseg
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import fire
from tqdm import tqdm
import glob
from scipy.stats import mannwhitneyu, gaussian_kde

matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

DATA_DIR = "cleaned_weibo_cov"
OUTPUT_DIR = "analysis_results"

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 官方媒体user_id列表（从配置文件加载）
OFFICIAL_MEDIA_IDS = set()

# 省份代码到名称的映射
PROVINCE_CODE_TO_NAME = {
    "11": "北京",
    "12": "天津",
    "13": "河北",
    "14": "山西",
    "15": "内蒙古",
    "21": "辽宁",
    "22": "吉林",
    "23": "黑龙江",
    "31": "上海",
    "32": "江苏",
    "33": "浙江",
    "34": "安徽",
    "35": "福建",
    "36": "江西",
    "37": "山东",
    "41": "河南",
    "42": "湖北",
    "43": "湖南",
    "44": "广东",
    "45": "广西",
    "46": "海南",
    "50": "重庆",
    "51": "四川",
    "52": "贵州",
    "53": "云南",
    "54": "西藏",
    "61": "陕西",
    "62": "甘肃",
    "63": "青海",
    "64": "宁夏",
    "65": "新疆",
    "71": "中国台湾",
    "81": "中国香港",
    "82": "中国澳门",
}


# ============================================================================
# 工具函数
# ============================================================================


def convert_province_code(code):
    """将省份代码转换为省份名称"""
    if pd.isna(code):
        return None
    # 统一转换为字符串格式处理
    if isinstance(code, (int, float)):
        code_str = str(int(code))  # 去掉小数点
    else:
        code_str = str(code).strip()

    # 如果是编码，转换为名称
    if code_str in PROVINCE_CODE_TO_NAME:
        return PROVINCE_CODE_TO_NAME[code_str]
    # 如果已经是名称，直接返回
    return code_str


def get_gender_label(gender):
    """将性别代码转换为中文标签"""
    gender_map = {"m": "男", "f": "女", "男": "男", "女": "女"}
    return gender_map.get(gender, gender)


def get_gender_color(gender):
    """获取性别对应的颜色"""
    color_map = {"m": "#20AEE6", "f": "#ff7333", "男": "#20AEE6", "女": "#ff7333"}
    return color_map.get(gender, "#808080")  # 默认灰色


def load_official_media_ids():
    """从配置文件加载官方媒体ID"""
    global OFFICIAL_MEDIA_IDS
    try:
        from get_news_ids import load_news_user_ids

        # 直接使用字符串ID（因为r_user_id是字符串格式）
        OFFICIAL_MEDIA_IDS = load_news_user_ids()
        print(f"已加载 {len(OFFICIAL_MEDIA_IDS)} 个官方媒体账号ID")
    except ImportError:
        print("警告: 无法导入 get_news_ids 模块，请确保已生成新闻账号ID")
    except Exception as e:
        print(f"加载官方媒体ID时出错: {e}")


# 停用词（从utils导入）
try:
    from utils.utils import STOP_WORDS, sentence_cleaner
except ImportError:
    # 如果导入失败，使用基本停用词
    STOP_WORDS = {
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
        "转发",
        "微博",
    }

    def sentence_cleaner(sentence, src_type="weibo"):
        import re

        if pd.isna(sentence) or sentence == "":
            return ""
        sentence = str(sentence)
        sentence = sentence.replace(""", "").replace(""", "")
        sentence = sentence.replace("…", "")
        sentence = sentence.replace("点击链接查看更多->", "")
        results = re.compile(r"[a-zA-Z0-9.?/&=:_%,-~#《》]", re.S)
        sentence = re.sub(results, "", sentence)
        results2 = re.compile(r"[//@].*?[:]", re.S)
        sentence = re.sub(results2, "", sentence)
        sentence = sentence.replace("\n", " ").strip()
        return sentence


# ============================================================================
# 新闻词表构建
# ============================================================================


def extract_noun_verb_words(text):
    """从文本中提取名词和动词（动名词）"""
    if pd.isna(text) or text == "":
        return []

    cleaned_text = sentence_cleaner(text)
    if cleaned_text == "" or cleaned_text == "转发微博":
        return []

    # 使用jieba进行词性标注
    words = pseg.cut(cleaned_text)
    # 提取名词(n)和动词(v)，过滤停用词和单字
    meaningful_words = [
        word
        for word, flag in words
        if flag.startswith(("n", "v")) and len(word) > 1 and word not in STOP_WORDS
    ]
    return meaningful_words


def load_official_media_content(year):
    """加载官方媒体账号发布的内容"""
    print(f"\n开始加载 {year} 年官方媒体账号发布的内容...")

    if not OFFICIAL_MEDIA_IDS:
        load_official_media_ids()

    if not OFFICIAL_MEDIA_IDS:
        print("警告: 官方媒体ID列表为空")
        return None

    year_dir = os.path.join(DATA_DIR, str(year))
    if not os.path.exists(year_dir):
        print(f"未找到 {year} 年的数据目录: {year_dir}")
        return None

    pattern = os.path.join(year_dir, "*.parquet")
    parquet_files = sorted(glob.glob(pattern))

    if not parquet_files:
        print(f"未找到 {year} 年的parquet文件")
        return None

    print(f"找到 {len(parquet_files)} 个文件")

    all_texts = []
    all_segmented_texts = []

    for file_path in tqdm(parquet_files, desc="加载官方媒体内容"):
        try:
            df = pd.read_parquet(file_path, columns=["user_id", "weibo_content"])
            # 只保留官方媒体账号发布的内容
            # 将user_id转换为字符串进行比较
            official_df = df[df["user_id"].astype(str).isin(OFFICIAL_MEDIA_IDS)]

            for content in official_df["weibo_content"]:
                if pd.notna(content) and content != "":
                    words = extract_noun_verb_words(content)
                    if len(words) > 0:
                        all_texts.append(" ".join(words))
                        all_segmented_texts.append(words)
        except Exception as e:
            print(f"读取文件 {file_path} 失败: {e}")
            continue

    print(f"共加载 {len(all_texts)} 条官方媒体内容")
    return all_texts, all_segmented_texts


def build_news_vocabulary(texts, top_n=5000):
    """使用TF-IDF构建官方媒体核心新闻词表"""
    print(f"\n开始构建官方媒体核心新闻词表（保留前 {top_n} 词）...")

    if len(texts) == 0:
        print("警告: 没有文本数据")
        return None

    # 使用TF-IDF提取重要词汇
    text_strings = [" ".join(text) for text in texts if len(text) > 0]

    if len(text_strings) == 0:
        print("警告: 没有有效的文本数据")
        return None

    # 使用自定义analyzer，因为文本已经分词
    def analyzer(text):
        return text.split()

    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        token_pattern=None,  # 禁用默认的token pattern
        max_features=top_n,
        min_df=2,  # 至少出现在2个文档中
        max_df=0.95,  # 最多出现在95%的文档中
        lowercase=False,  # 中文不需要转小写
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(text_strings)
        feature_names = vectorizer.get_feature_names_out()

        # 计算每个词的平均TF-IDF分数
        word_scores = {}
        tfidf_array = tfidf_matrix.toarray()

        for idx, word in enumerate(feature_names):
            # 计算该词在所有文档中的平均TF-IDF分数
            avg_score = tfidf_array[:, idx].mean()
            word_scores[word] = avg_score

        # 按分数排序，选择top_n个词
        sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        top_words = [word for word, score in sorted_words[:top_n]]

        print(f"成功构建词表，包含 {len(top_words)} 个词")
        return set(top_words)

    except Exception as e:
        print(f"构建词表时出错: {e}")
        # 如果TF-IDF失败，使用词频作为备选方案
        print("使用词频作为备选方案...")
        word_counter = Counter()
        for text in texts:
            if isinstance(text, list):
                word_counter.update(text)
            else:
                words = text.split()
                word_counter.update(words)

        top_words = [word for word, count in word_counter.most_common(top_n)]
        print(f"使用词频方法，包含 {len(top_words)} 个词")
        return set(top_words)


# ============================================================================
# Density计算函数
# ============================================================================


def calculate_news_density_fast(content, news_vocab):
    """快速计算news density（基于字符长度，不需要分词）

    直接检查news_vocab中的词在清理后的文本中的出现，
    计算占有的字符长度除以语句总长度。
    """
    if pd.isna(content) or content == "":
        return 0.0

    # 清理文本（删除链接、特殊字符等）
    cleaned_text = sentence_cleaner(content)
    if cleaned_text == "" or cleaned_text == "转发微博":
        return 0.0

    total_length = len(cleaned_text)
    if total_length == 0:
        return 0.0

    # 计算新闻词汇在文本中占有的字符长度
    news_char_length = 0
    for word in news_vocab:
        # 计算该词在文本中出现的所有位置的总字符长度
        count = cleaned_text.count(word)
        if count > 0:
            news_char_length += len(word) * count

    # news density = 新闻词汇字符长度 / 总字符长度
    density = news_char_length / total_length if total_length > 0 else 0.0
    return density


def calculate_post_density(year, news_vocab, force_recalculate=False):
    """计算post级别的news density（核心函数，只计算一次）

    返回包含所有post级别数据的DataFrame，列包括：
    - weibo_id: 微博ID
    - user_id: 用户ID
    - gender: 性别
    - province: 省份（已转换为名称）
    - density: news density值

    Args:
        year: 年份
        news_vocab: 新闻词表
        force_recalculate: 是否强制重新计算

    Returns:
        pd.DataFrame: post级别的density数据
    """
    post_density_file = os.path.join(OUTPUT_DIR, f"post_density_{year}.parquet")

    # 检查是否已有保存的数据
    if not force_recalculate and os.path.exists(post_density_file):
        print(f"\n找到已保存的post density数据: {post_density_file}")
        print("正在加载...")
        try:
            post_density_df = pd.read_parquet(post_density_file, engine="fastparquet")
            print(f"✓ 成功加载 {len(post_density_df):,} 条post数据")
            for gender in post_density_df["gender"].unique():
                count = len(post_density_df[post_density_df["gender"] == gender])
                print(f"  {gender}: {count:,} 条数据")
            return post_density_df
        except Exception as e:
            print(f"加载失败: {e}")
            print("将重新计算...")

    # 重新计算
    print(f"\n开始计算 {year} 年post级别的news density...")

    year_dir = os.path.join(DATA_DIR, str(year))
    if not os.path.exists(year_dir):
        print(f"未找到 {year} 年的数据目录: {year_dir}")
        return None

    pattern = os.path.join(year_dir, "*.parquet")
    parquet_files = sorted(glob.glob(pattern))

    if not parquet_files:
        print(f"未找到 {year} 年的parquet文件")
        return None

    print(f"找到 {len(parquet_files)} 个文件")

    # 存储所有DataFrame（分批处理，减少内存占用）
    df_list = []

    for file_path in tqdm(parquet_files, desc="计算news density"):
        try:
            # 读取必要的列
            required_columns = [
                "weibo_id",
                "user_id",
                "weibo_content",
                "gender",
                "province",
            ]
            df = pd.read_parquet(file_path, columns=required_columns)

            # 排除官方媒体用户
            df = df[~df["user_id"].astype(str).isin(OFFICIAL_MEDIA_IDS)]

            # 只保留有性别信息的记录
            df = df[df["gender"].notna()]

            if len(df) == 0:
                continue

            # 计算density
            df["density"] = df["weibo_content"].apply(
                lambda x: calculate_news_density_fast(x, news_vocab)
            )

            # 转换省份代码为省份名称
            df["province"] = df["province"].apply(convert_province_code)

            # 只保留需要的列
            df = df[["weibo_id", "user_id", "gender", "province", "density"]].copy()

            if len(df) > 0:
                df_list.append(df)

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            continue

    if not df_list:
        print("没有有效数据")
        return None

    # 合并所有DataFrame
    result_df = pd.concat(df_list, ignore_index=True)
    # dedup weibo_id
    result_df = result_df.drop_duplicates(subset=["weibo_id"], keep="first")
    print(f"共处理 {len(result_df):,} 条post数据")

    # 保存计算结果
    print(f"\n保存post density数据到: {post_density_file}")
    result_df.to_parquet(post_density_file, engine="fastparquet", index=False)
    print("✓ 保存成功")

    return result_df


def calculate_user_density_from_post(post_density_df):
    """从post级别的DataFrame计算user级别的平均density

    Args:
        post_density_df: post级别的DataFrame

    Returns:
        pd.DataFrame: user级别的density数据，包含user_id, gender, province, avg_density
    """
    print(f"\n从post级别数据计算user级别density...")

    if post_density_df is None or len(post_density_df) == 0:
        print("没有有效的post density数据")
        return None

    # 按user_id和gender聚合，计算每个用户的平均density
    user_density = (
        post_density_df.groupby(["user_id", "gender"])["density"].mean().reset_index()
    )
    user_density.columns = ["user_id", "gender", "avg_density"]

    # 如果有province信息，也添加到用户级别数据中（取每个用户最常见的省份）
    user_province = (
        post_density_df.groupby(["user_id", "gender"])["province"]
        .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None)
        .reset_index()
    )
    user_province.columns = ["user_id", "gender", "province"]
    user_density = user_density.merge(
        user_province, on=["user_id", "gender"], how="left"
    )

    print(f"共 {len(user_density):,} 个用户")

    return user_density


# ============================================================================
# 统计函数
# ============================================================================


def calculate_post_level_stats(post_density_df, year):
    """计算post级别的统计信息

    Args:
        post_density_df: DataFrame，包含weibo_id, user_id, gender, province, density列
        year: 年份

    Returns:
        list: 统计信息字典列表
    """
    stats = []

    if post_density_df is None or len(post_density_df) == 0:
        return stats

    for gender in post_density_df["gender"].unique():
        gender_df = post_density_df[post_density_df["gender"] == gender]

        if len(gender_df) == 0:
            continue

        densities = gender_df["density"].values

        # 计算0 density占比
        zero_count = np.sum(densities == 0.0)
        zero_ratio = zero_count / len(densities) if len(densities) > 0 else 0.0

        # 计算非0 density的统计
        non_zero_densities = densities[densities > 0.0]

        if len(non_zero_densities) > 0:
            avg_density = np.mean(non_zero_densities)
            median_density = np.median(non_zero_densities)
            std_density = np.std(non_zero_densities)
        else:
            avg_density = 0.0
            median_density = 0.0
            std_density = 0.0

        stats.append(
            {
                "gender": gender,
                "count": len(densities),
                "zero_count": int(zero_count),
                "zero_ratio": zero_ratio,
                "non_zero_count": len(non_zero_densities),
                "mean": avg_density,
                "median": median_density,
                "std": std_density,
            }
        )

    return stats


def calculate_user_level_stats(user_density_df, year):
    """计算user级别的统计信息

    Args:
        user_density_df: DataFrame，包含user_id, gender, avg_density列
        year: 年份

    Returns:
        list: 统计信息字典列表
    """
    stats = []

    if user_density_df is None or len(user_density_df) == 0:
        return stats

    for gender in user_density_df["gender"].unique():
        gender_data = user_density_df[user_density_df["gender"] == gender]
        total_users = len(gender_data)

        # 统计0 density用户
        zero_density_users = len(gender_data[gender_data["avg_density"] == 0.0])
        zero_density_ratio = (
            zero_density_users / total_users if total_users > 0 else 0.0
        )

        # 非0 density用户
        non_zero_data = gender_data[gender_data["avg_density"] > 0.0]
        non_zero_users = len(non_zero_data)

        if non_zero_users > 0:
            avg_density = non_zero_data["avg_density"].mean()
            median_density = non_zero_data["avg_density"].median()
            std_density = non_zero_data["avg_density"].std()
        else:
            avg_density = 0.0
            median_density = 0.0
            std_density = 0.0

        stats.append(
            {
                "gender": gender,
                "count": total_users,
                "zero_count": zero_density_users,
                "zero_ratio": zero_density_ratio,
                "non_zero_count": non_zero_users,
                "mean": avg_density,
                "median": median_density,
                "std": std_density,
            }
        )

    return stats


def calculate_province_stats(
    post_density_df, user_density_df, year, force_reanalyze=False
):
    """计算省份级别的统计信息（Post和User级别）

    Args:
        post_density_df: post级别的DataFrame
        user_density_df: user级别的DataFrame
        year: 年份
        force_reanalyze: 是否强制重新分析

    Returns:
        tuple: (province_post_stats_df, province_user_stats_df)
    """
    print(f"\n开始计算 {year} 年省份级别统计...")

    province_post_file = os.path.join(
        OUTPUT_DIR, f"province_post_density_{year}.parquet"
    )
    province_user_file = os.path.join(
        OUTPUT_DIR, f"province_user_density_{year}.parquet"
    )

    # 检查是否已有保存的数据
    if (
        not force_reanalyze
        and os.path.exists(province_post_file)
        and os.path.exists(province_user_file)
    ):
        print("找到已保存的省份级别数据，正在加载...")
        try:
            province_post_df = pd.read_parquet(province_post_file, engine="fastparquet")
            province_user_df = pd.read_parquet(province_user_file, engine="fastparquet")
            print("✓ 成功加载省份级别数据")
            return province_post_df, province_user_df
        except Exception as e:
            print(f"加载失败: {e}，将重新计算...")

    # 过滤有效省份
    valid_provinces = set(PROVINCE_CODE_TO_NAME.values())

    # Post级别统计
    post_with_province = post_density_df[
        (post_density_df["province"].notna())
        & (post_density_df["province"].isin(valid_provinces))
    ].copy()

    province_post_stats = []
    for province in sorted(post_with_province["province"].unique()):
        province_data = post_with_province[post_with_province["province"] == province]

        for gender in ["m", "f", "男", "女"]:
            gender_data = province_data[province_data["gender"] == gender]

            if len(gender_data) == 0:
                continue

            total_count = len(gender_data)
            zero_count = len(gender_data[gender_data["density"] == 0.0])
            zero_ratio = zero_count / total_count if total_count > 0 else 0.0

            # 计算非0 density的统计
            non_zero_data = gender_data[gender_data["density"] > 0.0]
            non_zero_count = len(non_zero_data)

            if non_zero_count > 0:
                avg_density = non_zero_data["density"].mean()
                std_density = non_zero_data["density"].std()
                # 计算99%置信区间
                z_99 = 2.576
                se = std_density / np.sqrt(non_zero_count)
                ci_lower = max(0, avg_density - z_99 * se)
                ci_upper = avg_density + z_99 * se
            else:
                avg_density = 0.0
                ci_lower = 0.0
                ci_upper = 0.0

            province_post_stats.append(
                {
                    "province": province,
                    "gender": gender,
                    "total_count": total_count,
                    "zero_count": zero_count,
                    "zero_ratio": zero_ratio,
                    "non_zero_count": non_zero_count,
                    "avg_density": avg_density,
                    "avg_density_ci_lower": ci_lower,
                    "avg_density_ci_upper": ci_upper,
                }
            )

    province_post_df = pd.DataFrame(province_post_stats)

    # User级别统计
    if user_density_df is not None and "province" in user_density_df.columns:
        user_with_province = user_density_df[
            (user_density_df["province"].notna())
            & (user_density_df["province"].isin(valid_provinces))
        ].copy()

        province_user_stats = []
        for province in sorted(user_with_province["province"].unique()):
            province_data = user_with_province[
                user_with_province["province"] == province
            ]

            for gender in ["m", "f", "男", "女"]:
                gender_data = province_data[province_data["gender"] == gender]

                if len(gender_data) == 0:
                    continue

                total_users = len(gender_data)
                zero_users = len(gender_data[gender_data["avg_density"] == 0.0])
                zero_ratio = zero_users / total_users if total_users > 0 else 0.0

                # 计算非0 density的统计
                non_zero_data = gender_data[gender_data["avg_density"] > 0.0]
                non_zero_users = len(non_zero_data)

                if non_zero_users > 0:
                    avg_density = non_zero_data["avg_density"].mean()
                    std_density = non_zero_data["avg_density"].std()
                    # 计算99%置信区间
                    z_99 = 2.576
                    se = std_density / np.sqrt(non_zero_users)
                    ci_lower = max(0, avg_density - z_99 * se)
                    ci_upper = avg_density + z_99 * se
                else:
                    avg_density = 0.0
                    ci_lower = 0.0
                    ci_upper = 0.0

                province_user_stats.append(
                    {
                        "province": province,
                        "gender": gender,
                        "total_users": total_users,
                        "zero_users": zero_users,
                        "zero_ratio": zero_ratio,
                        "non_zero_users": non_zero_users,
                        "avg_density": avg_density,
                        "avg_density_ci_lower": ci_lower,
                        "avg_density_ci_upper": ci_upper,
                    }
                )

        province_user_df = pd.DataFrame(province_user_stats)
    else:
        province_user_df = pd.DataFrame()

    # 保存结果
    province_post_df.to_parquet(province_post_file, engine="fastparquet", index=False)
    print(f"省份Post级别统计已保存到: {province_post_file}")

    if len(province_user_df) > 0:
        province_user_df.to_parquet(
            province_user_file, engine="fastparquet", index=False
        )
        print(f"省份User级别统计已保存到: {province_user_file}")

    return province_post_df, province_user_df


def print_stats(stats, year, level="post"):
    """打印统计信息（post和user级别共用）

    Args:
        stats: 统计信息字典列表
        year: 年份
        level: "post" 或 "user"
    """
    level_label = "Post级别" if level == "post" else "User级别"
    count_label = "有效内容数" if level == "post" else "总用户数"
    zero_label = "Density为0的数量" if level == "post" else "Density为0的用户数"
    ratio_label = "Density为0的占比" if level == "post" else "Density为0的用户占比"
    non_zero_label = "非0 Density数量" if level == "post" else "非0 Density用户数"

    print(f"\n{'='*60}")
    print(f"{level_label} News Density统计结果")
    print(f"{'='*60}")

    for stat in stats:
        gender = stat["gender"]
        gender_label = get_gender_label(gender)

        print(f"\n{gender_label}性 (代码: {gender}):")
        print(f"  {count_label}: {stat['count']:,} {'条' if level == 'post' else '人'}")
        print(
            f"  {zero_label}: {stat['zero_count']:,} {'条' if level == 'post' else '人'}"
        )
        print(
            f"  {ratio_label}: {stat['zero_ratio']:.4f} ({stat['zero_ratio']*100:.2f}%)"
        )
        print(
            f"  {non_zero_label}: {stat['non_zero_count']:,} {'条' if level == 'post' else '人'}"
        )

        if stat["non_zero_count"] > 0:
            print(f"  非0平均Density: {stat['mean']:.4f}")
            print(f"  非0中位Density: {stat['median']:.4f}")
            print(f"  非0标准差: {stat['std']:.4f}")

    # 保存统计结果
    if stats:
        stats_df = pd.DataFrame(stats)
        stats_file = os.path.join(
            OUTPUT_DIR,
            f"{'news_density' if level == 'post' else 'user_density'}_stats_{year}.parquet",
        )
        stats_df.to_parquet(stats_file, engine="fastparquet", index=False)
        print(f"\n统计结果已保存到: {stats_file}")


# ============================================================================
# 可视化函数
# ============================================================================


def _extract_gender_densities_from_df(df, density_col="density"):
    """从DataFrame中提取按性别分组的density列表（用于兼容旧的可视化函数）

    Args:
        df: DataFrame，包含gender和density列
        density_col: density列的名称

    Returns:
        dict: 键为性别，值为density列表
    """
    gender_densities = {}
    for gender in df["gender"].unique():
        gender_df = df[df["gender"] == gender]
        gender_densities[gender] = gender_df[density_col].tolist()
    return gender_densities


def visualize_density_distribution(data, year, level="post", n_plot_points=1000):
    """可视化news density的分布差异（post和user级别共用）

    绘制2x2布局的4个图：
    1. KDE密度图（对数X轴）
    2. 缺口箱线图（对数Y轴）
    3. 均值置信区间对比
    4. ECDF累积分布与统计检验

    Args:
        data: DataFrame（包含gender和density列）或字典（键为性别，值为density列表）
        year: 年份
        level: "post" 或 "user"
        n_plot_points: 绘制曲线时使用的点数，避免PDF文件过大（默认1000）
    """
    level_label = "Post级别" if level == "post" else "User级别"
    density_label = "News Density" if level == "post" else "用户平均News Density"

    print(f"\n开始绘制{level_label} news density分布对比图...")

    # 如果输入是DataFrame，转换为字典格式
    if isinstance(data, pd.DataFrame):
        density_col = "density" if level == "post" else "avg_density"
        gender_densities = _extract_gender_densities_from_df(data, density_col)
    else:
        gender_densities = data

    if not gender_densities:
        print("警告: 没有数据可绘制")
        return

    # 准备数据：只使用density>0的数据，保留全部数据用于拟合
    all_data_dict = {}  # 存储全部非零数据，用于拟合
    plot_data = []  # 用于箱线图等需要原始数据的图

    for gender, densities in gender_densities.items():
        densities_array = np.array(densities)
        # 过滤掉density为0的数据
        non_zero_densities = densities_array[densities_array > 0.0]
        n_samples = len(non_zero_densities)

        if n_samples == 0:
            print(f"  警告: {gender}性没有density>0的数据，跳过")
            continue

        # 保存全部非零数据用于拟合
        all_data_dict[gender] = non_zero_densities

        # 对于箱线图等，使用全部数据
        for density in non_zero_densities:
            plot_data.append({"gender": gender, "news_density": density})

    if len(plot_data) == 0:
        print("警告: 没有有效数据")
        return

    df_plot = pd.DataFrame(plot_data)
    # 将gender代码转换为中文标签
    df_plot["Gender"] = df_plot["gender"].apply(get_gender_label)
    df_plot["NewsDensity"] = df_plot["news_density"]

    total_non_zero = sum(len(data) for data in all_data_dict.values())
    print(f"  用于拟合的数据点总数: {total_non_zero:,}")
    print(f"  用于绘制的数据点总数: {len(df_plot):,}")

    # 构建颜色字典（使用中文标签）
    genders = df_plot["gender"].unique()
    gender_labels = [get_gender_label(g) for g in genders]
    my_palette = {get_gender_label(g): get_gender_color(g) for g in genders}

    # 创建2x2画布
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    fig.suptitle(
        f"{year}年{level_label} {density_label}分布差异综合分析",
        fontsize=18,
        fontweight="bold",
    )

    # 图1: KDE 密度图 (Log X轴)
    ax1 = axes[0, 0]
    all_values = np.concatenate([data for data in all_data_dict.values()])
    x_min_log = np.log10(all_values.min())
    x_max_log = np.log10(all_values.max())
    x_plot_log = np.linspace(x_min_log, x_max_log, n_plot_points)
    x_plot = 10**x_plot_log

    for gender in genders:
        gender_label = get_gender_label(gender)
        gender_color = get_gender_color(gender)
        data = all_data_dict[gender]

        kde = gaussian_kde(np.log10(data))
        kde_values = kde(x_plot_log)

        ax1.fill_between(
            x_plot, kde_values, alpha=0.3, color=gender_color, label=gender_label
        )
        ax1.plot(x_plot, kde_values, linewidth=2, color=gender_color)

    ax1.set_xscale("log")
    ax1.set_title("1. 整体分布形态 (KDE, Log X轴)", fontsize=14)
    ax1.set_xlabel(density_label, fontsize=12)
    ax1.set_ylabel("密度 (Density)", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 图2: 缺口箱线图 (Log Y轴)
    ax2 = axes[0, 1]
    sns.boxplot(
        data=df_plot,
        x="Gender",
        y="NewsDensity",
        palette=my_palette,
        notch=True,
        width=0.4,
        showfliers=False,
        ax=ax2,
    )
    ax2.set_yscale("log")
    ax2.set_title("2. 中位数差异检测 (Notched Boxplot)", fontsize=14)
    ax2.set_xlabel("性别", fontsize=12)
    ax2.set_ylabel(density_label, fontsize=12)
    ax2.grid(True, alpha=0.3, axis="y")

    # 图3: 均值置信区间对比
    ax3 = axes[1, 0]
    stats = (
        df_plot.groupby("Gender")["NewsDensity"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    stats["se"] = stats["std"] / np.sqrt(stats["count"])
    stats["ci99"] = 2.576 * stats["se"]  # 99% 置信区间半径

    x_coords = range(len(stats))
    categories = stats["Gender"].tolist()

    ax3.errorbar(
        x=x_coords,
        y=stats["mean"],
        yerr=stats["ci99"],
        fmt="none",
        ecolor="black",
        capsize=8,
        elinewidth=2,
    )

    for i, row in stats.iterrows():
        ax3.scatter(
            x=i,
            y=row["mean"],
            s=150,
            color=my_palette[row["Gender"]],
            zorder=5,
        )

    ax3.set_xticks(x_coords)
    ax3.set_xticklabels(categories, fontsize=12)
    ax3.set_title("3. 均值差异对比 (Mean + 99% CI)", fontsize=14)
    ax3.set_xlabel("性别", fontsize=12)
    ax3.set_ylabel(f"Mean {density_label}", fontsize=12)
    y_range = stats["mean"].max() - stats["mean"].min()
    if y_range > 0:
        ax3.set_ylim(
            stats["mean"].min() - y_range * 0.2,
            stats["mean"].max() + y_range * 0.2,
        )
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=my_palette[label], label=label) for label in categories
    ]
    ax3.legend(handles=legend_elements, loc="best")
    ax3.grid(True, alpha=0.3, axis="y")

    # 图4: ECDF + Mann-Whitney U 检验
    ax4 = axes[1, 1]

    for gender in genders:
        gender_label = get_gender_label(gender)
        gender_color = get_gender_color(gender)
        data = all_data_dict[gender]

        sorted_data = np.sort(data)
        ecdf_values = np.searchsorted(sorted_data, x_plot, side="right") / len(
            sorted_data
        )

        ax4.plot(
            x_plot, ecdf_values, linewidth=2, color=gender_color, label=gender_label
        )

    ax4.set_xscale("log")
    ax4.set_title("4. 累积分布 (ECDF) 与 统计检验", fontsize=14)
    ax4.set_xlabel(density_label, fontsize=12)
    ax4.set_ylabel("累积比例 (Proportion)", fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 执行非参数检验（Mann-Whitney U Test）
    if len(gender_labels) == 2:
        group1_label = gender_labels[0]
        group2_label = gender_labels[1]
        group1_gender = [g for g in genders if get_gender_label(g) == group1_label][0]
        group2_gender = [g for g in genders if get_gender_label(g) == group2_label][0]
        group1_data = all_data_dict[group1_gender]
        group2_data = all_data_dict[group2_gender]

        if len(group1_data) > 0 and len(group2_data) > 0:
            stat, p_val = mannwhitneyu(
                group1_data, group2_data, alternative="two-sided"
            )

            if p_val < 0.001:
                p_text = "P < 0.001 (极显著)"
            else:
                p_text = f"P = {p_val:.4f}"

            props = dict(
                boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"
            )
            ax4.text(
                0.05,
                0.15,
                f"Mann-Whitney U Test:\n"
                f"------------------\n"
                f"结果: {p_text}\n"
                f"结论: {'男性news density显著更高' if p_val < 0.05 else '分布无显著差异'}",
                transform=ax4.transAxes,
                fontsize=11,
                bbox=props,
                verticalalignment="bottom",
            )

    fig_filename = f"{'news_density' if level == 'post' else 'user_density'}_distribution_{year}.pdf"
    fig_path = os.path.join(OUTPUT_DIR, fig_filename)
    plt.savefig(fig_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()

    file_size_mb = os.path.getsize(fig_path) / (1024 * 1024)
    print(f"\n✓ {level_label}分布图已保存到: {fig_path}")
    print(f"  文件大小: {file_size_mb:.2f} MB")


def visualize_province_density_comparison(year):
    """绘制省份级别的News Density对比图（4个子图）

    1. Post级别0 density比例
    2. Post级别平均density（>0，带99% CI）
    3. User级别0 density比例
    4. User级别平均density（>0，带99% CI）
    """
    print(f"\n开始绘制 {year} 年省份级别News Density对比图...")

    province_post_file = os.path.join(
        OUTPUT_DIR, f"province_post_density_{year}.parquet"
    )
    province_user_file = os.path.join(
        OUTPUT_DIR, f"province_user_density_{year}.parquet"
    )

    if not os.path.exists(province_post_file):
        print(f"错误: 省份Post级别数据不存在: {province_post_file}")
        return

    province_post_df = pd.read_parquet(province_post_file, engine="fastparquet")

    if os.path.exists(province_user_file):
        province_user_df = pd.read_parquet(province_user_file, engine="fastparquet")
    else:
        province_user_df = pd.DataFrame()
        print("警告: 省份User级别数据不存在，将只绘制Post级别图表")

    # 统一性别标签
    province_post_df["gender_label"] = province_post_df["gender"].apply(
        get_gender_label
    )
    if len(province_user_df) > 0:
        province_user_df["gender_label"] = province_user_df["gender"].apply(
            get_gender_label
        )

    # 准备绘图数据
    provinces = sorted(province_post_df["province"].unique())
    plot_data = []

    for province in provinces:
        post_data = province_post_df[province_post_df["province"] == province]
        for gender in ["男", "女"]:
            gender_post = post_data[post_data["gender_label"] == gender]
            if len(gender_post) == 0:
                continue

            row = gender_post.iloc[0]
            plot_row = {
                "province": province,
                "gender": gender,
                "post_zero_ratio": row["zero_ratio"],
                "post_avg_density": (
                    row["avg_density"] if row["non_zero_count"] > 0 else 0
                ),
                "post_ci_lower": (
                    row["avg_density_ci_lower"] if row["non_zero_count"] > 0 else 0
                ),
                "post_ci_upper": (
                    row["avg_density_ci_upper"] if row["non_zero_count"] > 0 else 0
                ),
            }

            if len(province_user_df) > 0:
                user_data = province_user_df[province_user_df["province"] == province]
                gender_user = user_data[user_data["gender_label"] == gender]
                if len(gender_user) > 0:
                    user_row = gender_user.iloc[0]
                    plot_row["user_zero_ratio"] = user_row["zero_ratio"]
                    plot_row["user_avg_density"] = (
                        user_row["avg_density"] if user_row["non_zero_users"] > 0 else 0
                    )
                    plot_row["user_ci_lower"] = (
                        user_row["avg_density_ci_lower"]
                        if user_row["non_zero_users"] > 0
                        else 0
                    )
                    plot_row["user_ci_upper"] = (
                        user_row["avg_density_ci_upper"]
                        if user_row["non_zero_users"] > 0
                        else 0
                    )
                else:
                    plot_row["user_zero_ratio"] = 0
                    plot_row["user_avg_density"] = 0
                    plot_row["user_ci_lower"] = 0
                    plot_row["user_ci_upper"] = 0
            else:
                plot_row["user_zero_ratio"] = 0
                plot_row["user_avg_density"] = 0
                plot_row["user_ci_lower"] = 0
                plot_row["user_ci_upper"] = 0

            plot_data.append(plot_row)

    plot_df = pd.DataFrame(plot_data)

    if len(plot_df) == 0:
        print("没有足够的数据用于绘图")
        return

    # 创建4个子图
    num_provinces = len(provinces)
    fig_width = max(24, num_provinces * 0.8)
    fig, axes = plt.subplots(2, 2, figsize=(fig_width, 16), constrained_layout=True)
    fig.suptitle(
        f"{year}年各省份News Density性别差异分析", fontsize=18, fontweight="bold"
    )

    x_pos = np.arange(len(provinces))
    width = 0.35

    # 1. Post级别0 density比例
    ax1 = axes[0, 0]
    for gender in ["男", "女"]:
        gender_data = plot_df[plot_df["gender"] == gender]
        if len(gender_data) == 0:
            continue
        ratios = []
        for province in provinces:
            prov_data = gender_data[gender_data["province"] == province]
            if len(prov_data) > 0:
                ratios.append(prov_data.iloc[0]["post_zero_ratio"])
            else:
                ratios.append(0)

        offset = -width / 2 if gender == "男" else width / 2
        ax1.bar(
            x_pos + offset,
            ratios,
            width,
            label=gender,
            color=get_gender_color(gender),
            alpha=0.8,
        )

    ax1.set_xlabel("省份", fontsize=12)
    ax1.set_ylabel("0 Density比例", fontsize=12)
    ax1.set_title("1. Post级别0 Density比例", fontsize=14)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(provinces, rotation=45, ha="right", fontsize=8)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_ylim([0, 1.1])

    # 2. Post级别平均density（>0，带99% CI）
    ax2 = axes[0, 1]
    for gender in ["男", "女"]:
        gender_data = plot_df[plot_df["gender"] == gender]
        if len(gender_data) == 0:
            continue
        densities = []
        ci_lowers = []
        ci_uppers = []
        for province in provinces:
            prov_data = gender_data[gender_data["province"] == province]
            if len(prov_data) > 0 and prov_data.iloc[0]["post_avg_density"] > 0:
                densities.append(prov_data.iloc[0]["post_avg_density"])
                ci_lowers.append(prov_data.iloc[0]["post_ci_lower"])
                ci_uppers.append(prov_data.iloc[0]["post_ci_upper"])
            else:
                densities.append(0)
                ci_lowers.append(0)
                ci_uppers.append(0)

        offset = -width / 2 if gender == "男" else width / 2
        ax2.bar(
            x_pos + offset,
            densities,
            width,
            label=gender,
            color=get_gender_color(gender),
            alpha=0.8,
        )

        # 添加置信区间
        for i, (density, ci_l, ci_u) in enumerate(zip(densities, ci_lowers, ci_uppers)):
            if density > 0:
                ax2.errorbar(
                    i + offset,
                    density,
                    yerr=[[density - ci_l], [ci_u - density]],
                    fmt="none",
                    color="black",
                    capsize=3,
                    capthick=1,
                    alpha=0.6,
                )

    ax2.set_xlabel("省份", fontsize=12)
    ax2.set_ylabel("平均Density", fontsize=12)
    ax2.set_title("2. Post级别平均Density（>0，99% CI）", fontsize=14)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(provinces, rotation=45, ha="right", fontsize=8)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    # 3. User级别0 density比例
    ax3 = axes[1, 0]
    for gender in ["男", "女"]:
        gender_data = plot_df[plot_df["gender"] == gender]
        if len(gender_data) == 0:
            continue
        ratios = []
        for province in provinces:
            prov_data = gender_data[gender_data["province"] == province]
            if len(prov_data) > 0:
                ratios.append(prov_data.iloc[0]["user_zero_ratio"])
            else:
                ratios.append(0)

        offset = -width / 2 if gender == "男" else width / 2
        ax3.bar(
            x_pos + offset,
            ratios,
            width,
            label=gender,
            color=get_gender_color(gender),
            alpha=0.8,
        )

    ax3.set_xlabel("省份", fontsize=12)
    ax3.set_ylabel("0 Density比例", fontsize=12)
    ax3.set_title("3. User级别0 Density比例", fontsize=14)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(provinces, rotation=45, ha="right", fontsize=8)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.set_ylim([0, 1.1])

    # 4. User级别平均density（>0，带99% CI）
    ax4 = axes[1, 1]
    for gender in ["男", "女"]:
        gender_data = plot_df[plot_df["gender"] == gender]
        if len(gender_data) == 0:
            continue
        densities = []
        ci_lowers = []
        ci_uppers = []
        for province in provinces:
            prov_data = gender_data[gender_data["province"] == province]
            if len(prov_data) > 0 and prov_data.iloc[0]["user_avg_density"] > 0:
                densities.append(prov_data.iloc[0]["user_avg_density"])
                ci_lowers.append(prov_data.iloc[0]["user_ci_lower"])
                ci_uppers.append(prov_data.iloc[0]["user_ci_upper"])
            else:
                densities.append(0)
                ci_lowers.append(0)
                ci_uppers.append(0)

        offset = -width / 2 if gender == "男" else width / 2
        ax4.bar(
            x_pos + offset,
            densities,
            width,
            label=gender,
            color=get_gender_color(gender),
            alpha=0.8,
        )

        # 添加置信区间
        for i, (density, ci_l, ci_u) in enumerate(zip(densities, ci_lowers, ci_uppers)):
            if density > 0:
                ax4.errorbar(
                    i + offset,
                    density,
                    yerr=[[density - ci_l], [ci_u - density]],
                    fmt="none",
                    color="black",
                    capsize=3,
                    capthick=1,
                    alpha=0.6,
                )

    ax4.set_xlabel("省份", fontsize=12)
    ax4.set_ylabel("平均Density", fontsize=12)
    ax4.set_title("4. User级别平均Density（>0，99% CI）", fontsize=14)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(provinces, rotation=45, ha="right", fontsize=8)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis="y")

    # 保存图表
    fig_path = os.path.join(OUTPUT_DIR, f"province_density_comparison_{year}.pdf")
    plt.savefig(fig_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()
    print(f"图表已保存到: {fig_path}")
    print(f"\n省份级别News Density对比图绘制完成\n")


# ============================================================================
# 主入口函数
# ============================================================================


def analyze_news_density(year, force_recalculate=False, force_reanalyze=False):
    """News Density分析主入口函数

    统一的分析流程：
    1. 检查是否有已保存的post_density数据，如果没有则计算
    2. 基于post_density数据进行所有分析：
       - Post级别统计和可视化
       - User级别统计和可视化（从post级别聚合）
       - 省份级别统计和可视化（从post和user级别聚合）

    Args:
        year: 年份
        force_recalculate: 是否强制重新计算post_density（会重新计算所有数据）
        force_reanalyze: 是否强制重新分析省份级别数据（不会重新计算post_density）
    """
    print(f"\n{'='*60}")
    print(f"开始分析 {year} 年News Density（所有级别）")
    print(f"{'='*60}")

    # 加载或构建新闻词表
    news_vocab_file = os.path.join("configs", f"news_vocabulary_{year}.txt")
    if os.path.exists(news_vocab_file):
        with open(news_vocab_file, "r", encoding="utf-8") as f:
            news_vocab = set(line.strip() for line in f)
        print(f"✓ 已加载新闻词表，包含 {len(news_vocab)} 个词")
    else:
        print(f"未找到新闻词表文件: {news_vocab_file}")
        # 尝试构建新闻词表
        official_texts, official_segmented = load_official_media_content(year)
        if not official_segmented or len(official_segmented) == 0:
            print("错误: 无法加载官方媒体内容")
            return
        news_vocab = build_news_vocabulary(official_segmented, top_n=5000)
        if not news_vocab:
            print("错误: 无法构建新闻词表")
            return
        # 保存新闻词表
        vocab_file = os.path.join(OUTPUT_DIR, f"news_vocabulary_{year}.txt")
        with open(vocab_file, "w", encoding="utf-8") as f:
            for word in sorted(news_vocab):
                f.write(word + "\n")
        print(f"新闻词表已保存到: {vocab_file}")
        print("请人工检查")
        return

    # ========================================================================
    # 步骤1: 计算Post级别Density（核心步骤，只计算一次）
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"步骤1: 计算Post级别Density")
    print(f"{'='*60}")
    post_density_df = calculate_post_density(year, news_vocab, force_recalculate)
    if post_density_df is None or len(post_density_df) == 0:
        print("错误: 无法计算post density")
        return

    # ========================================================================
    # 步骤2: Post级别分析
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"步骤2: Post级别分析")
    print(f"{'='*60}")
    stats = calculate_post_level_stats(post_density_df, year)
    print_stats(stats, year, level="post")
    visualize_density_distribution(post_density_df, year, level="post")

    # ========================================================================
    # 步骤3: User级别分析（从Post级别聚合）
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"步骤3: User级别分析")
    print(f"{'='*60}")
    user_density_df = calculate_user_density_from_post(post_density_df)
    if user_density_df is not None and len(user_density_df) > 0:
        stats = calculate_user_level_stats(user_density_df, year)
        print_stats(stats, year, level="user")
        # 只可视化非0 density用户
        non_zero_user_density_df = user_density_df[
            user_density_df["avg_density"] > 0.0
        ].copy()
        if len(non_zero_user_density_df) > 0:
            visualize_density_distribution(non_zero_user_density_df, year, level="user")
        else:
            print("警告: 没有非0 density用户，无法绘制图表")
    else:
        print("警告: 无法计算user级别density")
        user_density_df = None

    # ========================================================================
    # 步骤4: 省份级别分析（从Post和User级别聚合）
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"步骤4: 省份级别分析")
    print(f"{'='*60}")
    province_post_df, province_user_df = calculate_province_stats(
        post_density_df, user_density_df, year, force_reanalyze
    )
    if province_post_df is not None and len(province_post_df) > 0:
        visualize_province_density_comparison(year)
    else:
        print("警告: 无法计算省份级别统计")

    print(f"\n{'='*60}")
    print(f"{year} 年News Density分析完成（所有级别）")
    print(f"{'='*60}")


if __name__ == "__main__":
    fire.Fire(
        {
            "analyze": analyze_news_density,
        }
    )
