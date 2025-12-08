"""
新闻密度分析脚本

从词汇角度分析男性和女性讨论内容与新闻官方媒体的重叠情况

功能：
1. 从官方媒体账号发布的内容中提取核心词汇（分词、提取动名词、TF-IDF，保留5000词）
2. 分析这些词在男性和女性发表内容中的出现频次，计算"news density"
3. 比较不同性别的平均news density和分布差异
"""

import os
import pandas as pd
import numpy as np
import jieba.posseg as pseg
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import fire
from tqdm import tqdm
import glob
import pickle

matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

DATA_DIR = "cleaned_weibo_cov"
OUTPUT_DIR = "analysis_results"

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 官方媒体user_id列表（从配置文件加载）
OFFICIAL_MEDIA_IDS = set()

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
        sentence = sentence.replace("“", "").replace("”", "")
        sentence = sentence.replace("…", "")
        sentence = sentence.replace("点击链接查看更多->", "")
        results = re.compile(r"[a-zA-Z0-9.?/&=:_%,-~#《》]", re.S)
        sentence = re.sub(results, "", sentence)
        results2 = re.compile(r"[//@].*?[:]", re.S)
        sentence = re.sub(results2, "", sentence)
        sentence = sentence.replace("\n", " ").strip()
        return sentence


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
    # texts是分词后的文本列表（每个元素是词列表）
    # 需要转换为字符串格式供TfidfVectorizer使用
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


def calculate_news_density(content, news_vocab):
    """计算单条内容的news density（新闻词汇占比）"""
    if pd.isna(content) or content == "":
        return 0.0

    words = extract_noun_verb_words(content)
    if len(words) == 0:
        return 0.0

    # 计算新闻词汇出现的次数
    news_word_count = sum(1 for word in words if word in news_vocab)

    # news density = 新闻词汇数 / 总词汇数
    density = news_word_count / len(words) if len(words) > 0 else 0.0
    return density


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


def analyze_news_density_by_gender(year, news_vocab):
    """分析不同性别的news density"""
    print(f"\n开始分析 {year} 年不同性别的news density...")

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

    # 使用字典存储每个性别的numpy数组
    gender_densities = {}

    for file_path in tqdm(parquet_files, desc="计算news density"):
        try:
            df = pd.read_parquet(
                file_path, columns=["user_id", "weibo_content", "gender"]
            )

            # 排除官方媒体用户
            df = df[~df["user_id"].astype(str).isin(OFFICIAL_MEDIA_IDS)]

            # 只保留有性别信息的记录
            df = df[df["gender"].notna()]

            # 按性别分组
            for gender in df["gender"].unique():
                gender_df = df[df["gender"] == gender]
                n_rows = len(gender_df)

                if n_rows == 0:
                    continue

                # 为该性别创建对应长度的numpy数组（全为NaN）
                if gender not in gender_densities:
                    # 如果是第一次遇到该性别，创建空列表用于后续合并
                    gender_densities[gender] = []

                # 创建当前批次的numpy数组
                arr = np.full(n_rows, np.nan, dtype=np.float64)

                # 按行计算并更新numpy数组对应值
                for idx, content in enumerate(gender_df["weibo_content"]):
                    arr[idx] = calculate_news_density_fast(content, news_vocab)

                # 将当前批次的数组添加到列表中
                gender_densities[gender].append(arr)

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            continue

    # 合并所有批次的数组
    result = {}
    for gender, arr_list in gender_densities.items():
        # 使用concatenate合并所有数组
        combined_arr = np.concatenate(arr_list)
        result[gender] = combined_arr.tolist()

    return result


def analyze_news_density_by_user_from_saved(year, gender_densities):
    """利用已保存的gender_densities进行用户级别分析（避免重新计算density）

    这个方法通过按照与analyze_news_density_by_gender完全相同的顺序重新读取文件，
    将已保存的density值分配给对应的user_id，从而避免重新计算。
    """
    print(f"\n利用已保存的density数据进行用户级别分析...")

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

    # 为每个性别创建density值的迭代器
    density_iterators = {}
    for gender, densities in gender_densities.items():
        density_iterators[gender] = iter(densities)

    # 存储所有DataFrame（分批处理，减少内存占用）
    df_list = []

    for file_path in tqdm(parquet_files, desc="匹配user_id和density"):
        try:
            df = pd.read_parquet(file_path, columns=["user_id", "gender"])

            # 排除官方媒体用户
            df = df[~df["user_id"].astype(str).isin(OFFICIAL_MEDIA_IDS)]

            # 只保留有性别信息的记录
            df = df[df["gender"].notna()]

            if len(df) == 0:
                continue

            # 初始化density列
            df["density"] = np.nan

            # 按性别分组（与analyze_news_density_by_gender保持相同的处理顺序）
            for gender in df["gender"].unique():
                gender_mask = df["gender"] == gender
                gender_df = df[gender_mask]

                if len(gender_df) == 0:
                    continue

                # 从对应性别的density迭代器中获取density值
                if gender not in density_iterators:
                    print(f"警告: 性别 {gender} 在已保存的density数据中不存在")
                    continue

                density_iter = density_iterators[gender]

                # 为每行分配对应的density值（直接在DataFrame上操作）
                density_values = []
                try:
                    for _ in range(len(gender_df)):
                        density_values.append(next(density_iter))
                except StopIteration:
                    print(f"警告: 性别 {gender} 的density值已用完，但还有数据行")
                    # 如果density值不够，用NaN填充剩余行
                    density_values.extend(
                        [np.nan] * (len(gender_df) - len(density_values))
                    )

                # 直接赋值给对应行
                df.loc[gender_mask, "density"] = density_values

            # 只保留需要的列，并过滤掉density为NaN的行（如果有的话）
            df = df[["user_id", "gender", "density"]].copy()
            df = df[df["density"].notna()]  # 移除density为NaN的行

            if len(df) > 0:
                df_list.append(df)

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            continue

    if not df_list:
        print("没有有效数据")
        return None

    # 合并所有DataFrame
    df_all = pd.concat(df_list, ignore_index=True)
    print(f"共处理 {len(df_all):,} 条post数据")

    # 按user_id和gender聚合，计算每个用户的平均density
    user_density = df_all.groupby(["user_id", "gender"])["density"].mean().reset_index()
    user_density.columns = ["user_id", "gender", "avg_density"]
    print(f"共 {len(user_density):,} 个用户")

    # 按性别分组存储
    user_densities_by_gender = {}
    for gender in user_density["gender"].unique():
        gender_data = user_density[user_density["gender"] == gender]
        user_densities_by_gender[gender] = gender_data["avg_density"].tolist()

    return user_densities_by_gender, user_density


def analyze_news_density_by_user(year, news_vocab):
    """分析用户级别的news density（计算每个用户的平均density）

    如果已保存了gender_densities，会优先使用analyze_news_density_by_user_from_saved来加速
    """
    print(f"\n开始分析 {year} 年用户级别的news density...")

    # 尝试使用已保存的gender_densities来加速
    gender_densities_file = os.path.join(OUTPUT_DIR, f"gender_densities_{year}.pkl")
    if os.path.exists(gender_densities_file):
        print(f"发现已保存的density数据，尝试利用其加速分析...")
        try:
            with open(gender_densities_file, "rb") as f:
                gender_densities = pickle.load(f)
            print(f"✓ 成功加载已保存的density数据")
            result = analyze_news_density_by_user_from_saved(year, gender_densities)
            if result is not None:
                return result
            else:
                print("利用已保存数据失败，回退到重新计算...")
        except Exception as e:
            print(f"加载已保存数据失败: {e}，回退到重新计算...")

    # 回退到原始方法：重新计算density
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

    for file_path in tqdm(parquet_files, desc="计算用户news density"):
        try:
            df = pd.read_parquet(
                file_path, columns=["user_id", "weibo_content", "gender"]
            )

            # 排除官方媒体用户
            df = df[~df["user_id"].astype(str).isin(OFFICIAL_MEDIA_IDS)]

            # 只保留有性别信息的记录
            df = df[df["gender"].notna()]

            if len(df) == 0:
                continue

            # 直接在DataFrame上计算density（使用apply，比iterrows快）
            df["density"] = df["weibo_content"].apply(
                lambda x: calculate_news_density_fast(x, news_vocab)
            )

            # 只保留需要的列
            df = df[["user_id", "gender", "density"]].copy()

            if len(df) > 0:
                df_list.append(df)

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            continue

    if not df_list:
        print("没有有效数据")
        return None

    # 合并所有DataFrame
    df_all = pd.concat(df_list, ignore_index=True)
    print(f"共处理 {len(df_all):,} 条post数据")

    # 按user_id和gender聚合，计算每个用户的平均density
    user_density = df_all.groupby(["user_id", "gender"])["density"].mean().reset_index()
    user_density.columns = ["user_id", "gender", "avg_density"]
    print(f"共 {len(user_density):,} 个用户")

    # 按性别分组存储
    user_densities_by_gender = {}
    for gender in user_density["gender"].unique():
        gender_data = user_density[user_density["gender"] == gender]
        user_densities_by_gender[gender] = gender_data["avg_density"].tolist()

    return user_densities_by_gender, user_density


def visualize_news_density_distribution(
    gender_densities, year, max_samples_per_gender=100000
):
    """可视化news density的分布差异

    Args:
        gender_densities: 字典，键为性别，值为density列表
        year: 年份
        max_samples_per_gender: 每个性别最多采样的数据点数，避免PDF文件过大
    """
    print(f"\n开始绘制news density分布对比图...")

    if not gender_densities:
        print("警告: 没有数据可绘制")
        return

    # 准备数据并进行采样，避免数据点过多导致PDF文件过大
    plot_data = []
    for gender, densities in gender_densities.items():
        densities_array = np.array(densities)
        n_samples = len(densities_array)

        # 如果数据点太多，进行随机采样
        if n_samples > max_samples_per_gender:
            print(
                f"  {gender}性: {n_samples:,} 个数据点，采样至 {max_samples_per_gender:,} 个"
            )
            sampled_indices = np.random.choice(
                n_samples, size=max_samples_per_gender, replace=False
            )
            sampled_densities = densities_array[sampled_indices]
        else:
            sampled_densities = densities_array

        for density in sampled_densities:
            plot_data.append({"gender": gender, "news_density": density})

    if len(plot_data) == 0:
        print("警告: 没有有效数据")
        return

    df_plot = pd.DataFrame(plot_data)
    print(f"  用于绘图的数据点总数: {len(df_plot):,}")

    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：箱线图
    ax1 = axes[0]
    genders = df_plot["gender"].unique()
    box_data = [df_plot[df_plot["gender"] == g]["news_density"].values for g in genders]
    bp = ax1.boxplot(box_data, labels=genders, patch_artist=True)
    colors = ["#FF6B6B", "#4ECDC4", "#95E1D3", "#F38181"]
    for patch, color in zip(bp["boxes"], colors[: len(bp["boxes"])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.set_xlabel("性别", fontsize=12)
    ax1.set_ylabel("News Density", fontsize=12)
    ax1.set_title(
        f"{year}年News Density分布对比（箱线图）",
        fontsize=13,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)

    # 右图：KDE图（使用采样后的数据）
    ax2 = axes[1]
    for gender in genders:
        gender_data = df_plot[df_plot["gender"] == gender]["news_density"]
        # 使用bw_adjust参数控制KDE的平滑度，避免过度密集的曲线
        # 注意：即使KDE画出来是一条线，但当数据点非常多时，
        # matplotlib的PDF后端会存储大量路径点，导致文件巨大
        # 因此需要对数据进行采样，或使用bw_adjust使曲线更平滑
        sns.kdeplot(
            gender_data,
            ax=ax2,
            label=f"{gender}性",
            linewidth=2,
            bw_adjust=0.5,  # 增加带宽调整，使曲线更平滑，减少PDF文件大小
            gridsize=200,  # 限制KDE评估的网格点数，进一步减少路径点数量
        )
    ax2.set_xlabel("News Density", fontsize=12)
    ax2.set_ylabel("密度", fontsize=12)
    ax2.set_title(
        f"{year}年News Density分布对比（KDE）",
        fontsize=13,
        fontweight="bold",
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, f"news_density_distribution_{year}.pdf")
    # 使用rasterized=True可以进一步减小PDF文件大小（将某些元素栅格化）
    plt.savefig(
        fig_path,
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()

    # 检查文件大小
    file_size_mb = os.path.getsize(fig_path) / (1024 * 1024)
    print(f"News Density分布对比图已保存到: {fig_path}")
    print(f"  文件大小: {file_size_mb:.2f} MB")


def analyze_news_density(year):
    """主分析函数"""
    print(f"\n{'='*60}")
    print(f"开始分析 {year} 年新闻密度（News Density）")
    print(f"{'='*60}")

    news_vocab_file = os.path.join("configs", f"news_vocabulary_{year}.txt")
    has_vocab = False

    if os.path.exists(news_vocab_file):
        has_vocab = True
        with open(news_vocab_file, "r", encoding="utf-8") as f:
            news_vocab = set(line.strip() for line in f)
    else:
        print(f"未找到新闻词表文件: {news_vocab_file}")

    if not has_vocab:
        # 1. 加载官方媒体内容
        official_texts, official_segmented = load_official_media_content(year)
        if not official_segmented or len(official_segmented) == 0:
            print("错误: 无法加载官方媒体内容")
            return

        # 2. 构建官方媒体核心新闻词表
        # 使用分词后的文本列表（词列表的列表）
        news_vocab = build_news_vocabulary(official_segmented, top_n=5000)
        if not news_vocab:
            print("错误: 无法构建新闻词表")
            return

        # 3. 保存新闻词表
        vocab_file = os.path.join(OUTPUT_DIR, f"news_vocabulary_{year}.txt")
        with open(vocab_file, "w", encoding="utf-8") as f:
            for word in sorted(news_vocab):
                f.write(word + "\n")
        print(f"新闻词表已保存到: {vocab_file}")
        print("请人工检查")
        return

    # 4. 分析不同性别的news density（检查是否已有保存的数据）
    gender_densities_file = os.path.join(OUTPUT_DIR, f"gender_densities_{year}.pkl")

    if os.path.exists(gender_densities_file):
        print(f"\n找到已保存的gender_densities数据: {gender_densities_file}")
        print("正在加载...")
        try:
            with open(gender_densities_file, "rb") as f:
                gender_densities = pickle.load(f)
            print(f"✓ 成功加载 {len(gender_densities)} 个性别的数据")
            for gender, densities in gender_densities.items():
                print(f"  {gender}: {len(densities):,} 条数据")
        except Exception as e:
            print(f"加载失败: {e}")
            print("将重新计算...")
            gender_densities = analyze_news_density_by_gender(year, news_vocab)
            if not gender_densities:
                print("错误: 无法计算news density")
                return
            # 保存计算结果
            print(f"\n保存gender_densities到: {gender_densities_file}")
            with open(gender_densities_file, "wb") as f:
                pickle.dump(gender_densities, f)
            print("✓ 保存成功")
    else:
        print(f"\n未找到已保存的数据，开始计算...")
        gender_densities = analyze_news_density_by_gender(year, news_vocab)
        if not gender_densities:
            print("错误: 无法计算news density")
            return
        # 保存计算结果
        print(f"\n保存gender_densities到: {gender_densities_file}")
        with open(gender_densities_file, "wb") as f:
            pickle.dump(gender_densities, f)
        print("✓ 保存成功")

    # 5. 打印统计信息
    print(f"\n{'='*60}")
    print(f"News Density统计结果")
    print(f"{'='*60}")

    stats = []
    for gender, densities in gender_densities.items():
        if len(densities) > 0:
            avg_density = np.mean(densities)
            median_density = np.median(densities)
            std_density = np.std(densities)
            stats.append(
                {
                    "gender": gender,
                    "count": len(densities),
                    "mean": avg_density,
                    "median": median_density,
                    "std": std_density,
                }
            )

            print(f"\n{gender}性统计:")
            print(f"  有效内容数: {len(densities):,} 条")
            print(f"  平均News Density: {avg_density:.4f}")
            print(f"  中位News Density: {median_density:.4f}")
            print(f"  标准差: {std_density:.4f}")

    # 保存统计结果
    if stats:
        stats_df = pd.DataFrame(stats)
        stats_file = os.path.join(OUTPUT_DIR, f"news_density_stats_{year}.parquet")
        stats_df.to_parquet(stats_file, engine="fastparquet", index=False)
        print(f"\n统计结果已保存到: {stats_file}")

    # 6. 可视化分布差异
    visualize_news_density_distribution(gender_densities, year)

    print(f"\n{'='*60}")
    print(f"{year} 年新闻密度分析完成")
    print(f"{'='*60}")


def analyze_user_level_news_density(year):
    """独立的用户级别News Density分析函数"""
    print(f"\n{'='*60}")
    print(f"开始用户级别的News Density分析")
    print(f"{'='*60}")

    # 检查是否需要news_vocab（如果已保存数据不存在，需要重新计算）
    user_densities_file = os.path.join(OUTPUT_DIR, f"user_densities_{year}.pkl")
    user_density_df_file = os.path.join(OUTPUT_DIR, f"user_density_df_{year}.parquet")

    news_vocab = None
    if not (
        os.path.exists(user_densities_file) and os.path.exists(user_density_df_file)
    ):
        # 需要重新计算，先加载news_vocab
        news_vocab_file = os.path.join("configs", f"news_vocabulary_{year}.txt")
        if os.path.exists(news_vocab_file):
            with open(news_vocab_file, "r", encoding="utf-8") as f:
                news_vocab = set(line.strip() for line in f)
        else:
            print(f"错误: 未找到新闻词表文件: {news_vocab_file}")
            print("请先运行 analyze 命令生成新闻词表")
            return

    # 检查是否已有保存的用户级别数据
    if os.path.exists(user_densities_file) and os.path.exists(user_density_df_file):
        print(f"\n找到已保存的用户级别数据")
        try:
            with open(user_densities_file, "rb") as f:
                user_densities_by_gender = pickle.load(f)
            user_density_df = pd.read_parquet(
                user_density_df_file, engine="fastparquet"
            )
            print(f"✓ 成功加载用户级别数据")
        except Exception as e:
            print(f"加载失败: {e}，将重新计算...")
            result = analyze_news_density_by_user(year, news_vocab)
            if result is None:
                print("错误: 无法计算用户级别news density")
                return
            else:
                user_densities_by_gender, user_density_df = result
                # 保存结果
                with open(user_densities_file, "wb") as f:
                    pickle.dump(user_densities_by_gender, f)
                user_density_df.to_parquet(
                    user_density_df_file, engine="fastparquet", index=False
                )
                print(f"✓ 用户级别数据已保存")
    else:
        # 需要重新计算
        result = analyze_news_density_by_user(year, news_vocab)
        if result is None:
            print("错误: 无法计算用户级别news density")
            return
        else:
            user_densities_by_gender, user_density_df = result
            # 保存结果
            with open(user_densities_file, "wb") as f:
                pickle.dump(user_densities_by_gender, f)
            user_density_df.to_parquet(
                user_density_df_file, engine="fastparquet", index=False
            )
            print(f"✓ 用户级别数据已保存")

    # 分析用户级别的统计
    analyze_user_level_stats(user_density_df, year)
    visualize_user_level_density(user_density_df, year)

    print(f"\n{'='*60}")
    print(f"{year} 年用户级别News Density分析完成")
    print(f"{'='*60}")


def analyze_user_level_stats(user_density_df, year):
    """分析用户级别的统计信息"""
    print(f"\n{'='*60}")
    print(f"用户级别News Density统计结果")
    print(f"{'='*60}")

    stats = []

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

        gender_label = get_gender_label(gender)

        stats.append(
            {
                "gender": gender,
                "total_users": total_users,
                "zero_density_users": zero_density_users,
                "zero_density_ratio": zero_density_ratio,
                "non_zero_users": non_zero_users,
                "avg_density": avg_density,
                "median_density": median_density,
                "std_density": std_density,
            }
        )

        print(f"\n{gender_label}性 (代码: {gender}):")
        print(f"  总用户数: {total_users:,} 人")
        print(f"  Density为0的用户数: {zero_density_users:,} 人")
        print(
            f"  Density为0的用户占比: {zero_density_ratio:.4f} ({zero_density_ratio*100:.2f}%)"
        )
        print(f"  非0 Density用户数: {non_zero_users:,} 人")
        if non_zero_users > 0:
            print(f"  非0用户平均Density: {avg_density:.4f}")
            print(f"  非0用户中位Density: {median_density:.4f}")
            print(f"  非0用户标准差: {std_density:.4f}")

    # 保存统计结果
    if stats:
        stats_df = pd.DataFrame(stats)
        stats_file = os.path.join(OUTPUT_DIR, f"user_density_stats_{year}.parquet")
        stats_df.to_parquet(stats_file, engine="fastparquet", index=False)
        print(f"\n用户级别统计结果已保存到: {stats_file}")


def visualize_user_level_density(user_density_df, year):
    """可视化用户级别的density分布（只包含非0 density用户）"""
    print(f"\n{'='*60}")
    print(f"生成用户级别News Density可视化图表")
    print(f"{'='*60}")

    # 过滤掉0 density用户
    non_zero_df = user_density_df[user_density_df["avg_density"] > 0.0].copy()

    if len(non_zero_df) == 0:
        print("没有非0 density用户，无法绘制图表")
        return

    # 按性别分组
    genders = non_zero_df["gender"].unique()
    if len(genders) < 2:
        print("警告: 数据不足，无法进行性别对比")
        return

    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：箱线图
    ax1 = axes[0]
    box_data = [
        non_zero_df[non_zero_df["gender"] == g]["avg_density"].values for g in genders
    ]
    gender_labels = [get_gender_label(g) for g in genders]
    bp = ax1.boxplot(box_data, labels=gender_labels, patch_artist=True)

    # 设置箱线图颜色
    for patch, gender in zip(bp["boxes"], genders):
        color = get_gender_color(gender)
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax1.set_xlabel("性别", fontsize=12)
    ax1.set_ylabel("用户平均News Density", fontsize=12)
    ax1.set_title(
        f"{year}年用户平均News Density分布对比（箱线图，Density > 0）",
        fontsize=13,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)

    # 右图：KDE图
    ax2 = axes[1]
    for gender in genders:
        gender_data = non_zero_df[non_zero_df["gender"] == gender]["avg_density"]
        gender_label = get_gender_label(gender)
        gender_color = get_gender_color(gender)
        sns.kdeplot(
            gender_data, ax=ax2, label=gender_label, linewidth=2, color=gender_color
        )

    ax2.set_xlabel("用户平均News Density", fontsize=12)
    ax2.set_ylabel("密度", fontsize=12)
    ax2.set_title(
        f"{year}年用户平均News Density分布对比（KDE，Density > 0）",
        fontsize=13,
        fontweight="bold",
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, f"user_density_distribution_{year}.pdf")
    plt.savefig(fig_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()

    file_size_mb = os.path.getsize(fig_path) / (1024 * 1024)
    print(f"\n✓ 用户级别分布图已保存到: {fig_path}")
    print(f"  文件大小: {file_size_mb:.2f} MB")


def get_gender_label(gender):
    """将性别代码转换为中文标签"""
    gender_map = {"m": "男", "f": "女", "男": "男", "女": "女"}
    return gender_map.get(gender, gender)


def get_gender_color(gender):
    """获取性别对应的颜色"""
    color_map = {"m": "#20AEE6", "f": "#ff7333", "男": "#20AEE6", "女": "#ff7333"}
    return color_map.get(gender, "#808080")  # 默认灰色


def visualize_from_saved_stats(year):
    """Generate boxplot-style visualization from saved statistics

    This function reads the saved statistics file and generates boxplot-style charts
    to visualize gender differences in news density.
    Note: Since we only have summary statistics (mean, median, std), we create
    a boxplot-style visualization based on these statistics.
    """
    print(f"\n{'='*60}")
    print(f"Generating News Density visualization for year {year} from saved data")
    print(f"{'='*60}")

    # 1. Load original density data to calculate zero density ratio
    gender_densities_file = os.path.join(OUTPUT_DIR, f"gender_densities_{year}.pkl")
    if not os.path.exists(gender_densities_file):
        print(f"Error: Original density data file not found: {gender_densities_file}")
        print("Please run analyze_news_density function first to generate data")
        return

    # Load original density data
    print(f"\nLoading original density data from: {gender_densities_file}")
    try:
        with open(gender_densities_file, "rb") as f:
            gender_densities = pickle.load(f)
        print(f"✓ Successfully loaded density data for {len(gender_densities)} genders")
    except Exception as e:
        print(f"Error loading density data: {e}")
        return

    # 2. Calculate zero density ratio for each gender
    print(f"\n{'='*60}")
    print(f"Zero Density Analysis")
    print(f"{'='*60}")
    zero_density_stats = []
    filtered_gender_densities = {}

    for gender, densities in gender_densities.items():
        if len(densities) == 0:
            continue

        total_count = len(densities)
        zero_count = sum(1 for d in densities if d == 0.0)
        zero_ratio = zero_count / total_count if total_count > 0 else 0.0

        gender_label = get_gender_label(gender)
        zero_density_stats.append(
            {
                "gender": gender,
                "gender_label": gender_label,
                "total_count": total_count,
                "zero_count": zero_count,
                "zero_ratio": zero_ratio,
            }
        )

        print(f"\n{gender_label}性 (代码: {gender}):")
        print(f"  总内容数: {total_count:,} 条")
        print(f"  Density为0的数量: {zero_count:,} 条")
        print(f"  Density为0的占比: {zero_ratio:.4f} ({zero_ratio*100:.2f}%)")

        # Filter out zero density values
        filtered_densities = [d for d in densities if d > 0.0]
        filtered_gender_densities[gender] = filtered_densities
        print(f"  过滤后（density>0）的数量: {len(filtered_densities):,} 条")

    # 3. Calculate statistics for filtered data (density > 0)
    print(f"\n{'='*60}")
    print(f"Statistics for Non-Zero Density")
    print(f"{'='*60}")

    stats = []
    for gender, densities in filtered_gender_densities.items():
        if len(densities) > 0:
            avg_density = np.mean(densities)
            median_density = np.median(densities)
            std_density = np.std(densities)
            stats.append(
                {
                    "gender": gender,
                    "count": len(densities),
                    "mean": avg_density,
                    "median": median_density,
                    "std": std_density,
                }
            )

            gender_label = get_gender_label(gender)
            print(f"\n{gender_label}性 (代码: {gender}) 统计（density>0）:")
            print(f"  有效内容数: {len(densities):,} 条")
            print(f"  平均News Density: {avg_density:.4f}")
            print(f"  中位News Density: {median_density:.4f}")
            print(f"  标准差: {std_density:.4f}")

    if len(stats) < 2:
        print("Warning: Insufficient data for gender comparison")
        return

    # 4. Create bar chart with error bars showing mean and standard deviation
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    stats_df = pd.DataFrame(stats)
    genders = stats_df["gender"].values
    means = stats_df["mean"].values
    stds = stats_df["std"].values

    # Generate gender labels and colors
    gender_labels = [get_gender_label(g) for g in genders]
    bar_colors = [get_gender_color(g) for g in genders]

    # Create bar chart with error bars
    positions = np.arange(1, len(genders) + 1)
    width = 0.6

    # Draw bars
    bars = ax.bar(
        positions,
        means,
        width=width,
        color=bar_colors,
        alpha=0.7,
        edgecolor="black",
        linewidth=1.5,
        label="Mean",
    )

    # Add error bars (standard deviation)
    ax.errorbar(
        positions,
        means,
        yerr=stds,
        fmt="none",
        color="black",
        capsize=8,
        capthick=2,
        linewidth=2,
        label="±1 SD",
    )

    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        # Label with mean value
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + std + 0.01 * max(means),
            f"{mean:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
        # Label with SD value
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + std / 2,
            f"SD: {std:.4f}",
            ha="center",
            va="center",
            fontsize=9,
            style="italic",
        )

    # Set labels and title
    ax.set_xticks(positions)
    ax.set_xticklabels(gender_labels, fontsize=12, fontweight="bold")
    ax.set_ylabel("News Density", fontsize=13, fontweight="bold")
    ax.set_title(
        f"News Density by Gender ({year}, Density > 0)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")

    # Ensure y-axis starts from 0 or slightly below
    y_min = max(0, min(means - stds) * 1.1)
    ax.set_ylim(bottom=y_min)

    # Add legend
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, f"news_density_boxplot_{year}.pdf")
    plt.savefig(fig_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()

    file_size_mb = os.path.getsize(fig_path) / (1024 * 1024)
    print(f"\n✓ Boxplot saved to: {fig_path}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"\n{'='*60}")
    print(f"Visualization completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    fire.Fire(
        {
            "analyze": analyze_news_density,
            "analyze_user": analyze_user_level_news_density,
            "visualize": visualize_from_saved_stats,
        }
    )
