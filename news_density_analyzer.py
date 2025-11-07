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

        raw_ids = load_news_user_ids()
        # 将字符串ID转换为整数（因为user_id通常是int64类型）
        # 同时保留字符串版本以兼容不同情况
        OFFICIAL_MEDIA_IDS = set()
        for id_str in raw_ids:
            try:
                # 尝试转换为整数
                id_int = int(id_str)
                OFFICIAL_MEDIA_IDS.add(id_int)
            except (ValueError, TypeError):
                # 如果转换失败，保留原字符串
                OFFICIAL_MEDIA_IDS.add(id_str)
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
            official_df = df[df["user_id"].isin(OFFICIAL_MEDIA_IDS)]

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

    # 排除官方媒体用户ID（避免官方媒体转发自己）
    gender_densities = defaultdict(list)

    for file_path in tqdm(parquet_files, desc="计算news density"):
        try:
            df = pd.read_parquet(
                file_path, columns=["user_id", "weibo_content", "gender"]
            )

            # 排除官方媒体用户
            df = df[~df["user_id"].isin(OFFICIAL_MEDIA_IDS)]

            # 只保留有性别信息的记录
            df = df[df["gender"].notna()]

            for _, row in df.iterrows():
                density = calculate_news_density(row["weibo_content"], news_vocab)
                # 记录所有内容的density（包括0），以便完整分析分布
                gender_densities[row["gender"]].append(density)

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            continue

    return gender_densities


def visualize_news_density_distribution(gender_densities, year):
    """可视化news density的分布差异"""
    print(f"\n开始绘制news density分布对比图...")

    if not gender_densities:
        print("警告: 没有数据可绘制")
        return

    # 准备数据
    plot_data = []
    for gender, densities in gender_densities.items():
        for density in densities:
            plot_data.append({"gender": gender, "news_density": density})

    if len(plot_data) == 0:
        print("警告: 没有有效数据")
        return

    df_plot = pd.DataFrame(plot_data)

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

    # 右图：KDE图
    ax2 = axes[1]
    for gender in genders:
        gender_data = df_plot[df_plot["gender"] == gender]["news_density"]
        sns.kdeplot(gender_data, ax=ax2, label=f"{gender}性", linewidth=2)
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
    plt.savefig(fig_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()
    print(f"News Density分布对比图已保存到: {fig_path}")


def analyze_news_density(year):
    """主分析函数"""
    print(f"\n{'='*60}")
    print(f"开始分析 {year} 年新闻密度（News Density）")
    print(f"{'='*60}")

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

    # 4. 分析不同性别的news density
    gender_densities = analyze_news_density_by_gender(year, news_vocab)
    if not gender_densities:
        print("错误: 无法计算news density")
        return

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


if __name__ == "__main__":
    fire.Fire(
        {
            "analyze": analyze_news_density,
        }
    )
