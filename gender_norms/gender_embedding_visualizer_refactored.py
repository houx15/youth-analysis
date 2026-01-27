"""
省份性别-职业偏向可视化分析（重构版）

按照 gender_embedding_visualizer_spec.md 的规范：
1. 职业与性别分析（基于相似度方法 + 投影方法）
2. 家务劳动与性别分析（基于相似度方法 + 投影方法）

使用 method 参数避免代码重复
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from matplotlib.font_manager import FontProperties
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

try:
    plt.rcParams["font.sans-serif"] = [
        "Arial Unicode MS",
        "SimHei",
        "STHeiti",
        "Microsoft YaHei",
    ]
except:
    pass

INPUT_DIR = "gender_embedding/results/embedding_analysis"
OUTPUT_DIR = "gender_embedding/results/embedding_visualization"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 省份编码映射
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
    "71": "台湾",
    "81": "香港",
    "82": "澳门",
}

PROVINCE_NAME_MAPPING = {
    "北京": "北京市",
    "天津": "天津市",
    "上海": "上海市",
    "重庆": "重庆市",
    "河北": "河北省",
    "山西": "山西省",
    "辽宁": "辽宁省",
    "吉林": "吉林省",
    "黑龙江": "黑龙江省",
    "江苏": "江苏省",
    "浙江": "浙江省",
    "安徽": "安徽省",
    "福建": "福建省",
    "江西": "江西省",
    "山东": "山东省",
    "河南": "河南省",
    "湖北": "湖北省",
    "湖南": "湖南省",
    "广东": "广东省",
    "海南": "海南省",
    "四川": "四川省",
    "贵州": "贵州省",
    "云南": "云南省",
    "陕西": "陕西省",
    "甘肃": "甘肃省",
    "青海": "青海省",
    "台湾": "台湾省",
    "内蒙古": "内蒙古自治区",
    "广西": "广西壮族自治区",
    "西藏": "西藏自治区",
    "宁夏": "宁夏回族自治区",
    "新疆": "新疆维吾尔自治区",
    "香港": "香港特别行政区",
    "澳门": "澳门特别行政区",
}

PROVINCE_REGIONS = {
    "华北": ["北京", "天津", "河北", "山西", "内蒙古"],
    "东北": ["辽宁", "吉林", "黑龙江"],
    "华东": ["上海", "江苏", "浙江", "安徽", "福建", "江西", "山东"],
    "华中": ["河南", "湖北", "湖南"],
    "华南": ["广东", "广西", "海南"],
    "西南": ["重庆", "四川", "贵州", "云南", "西藏"],
    "西北": ["陕西", "甘肃", "青海", "宁夏", "新疆"],
}

PROVINCE_TO_REGION = {}
for region, provinces in PROVINCE_REGIONS.items():
    for province in provinces:
        PROVINCE_TO_REGION[province] = region


def generate_output_path(filename, year):
    """
    生成带日期前缀的输出文件路径

    Args:
        filename: 基础文件名（不含扩展名）
        year: 分析年份

    Returns:
        完整的输出路径（带日期前缀和.pdf扩展名）

    Examples:
        >>> generate_output_path("occupation_map_similarity", 2020)
        "gender_embedding/results/embedding_visualization/20260106_occupation_map_similarity_2020.pdf"
    """
    date_prefix = datetime.now().strftime("%Y%m%d")
    output_filename = f"{date_prefix}_{filename}_{year}.pdf"
    return os.path.join(OUTPUT_DIR, output_filename)


def convert_province_code(province):
    """将省份编码转换为省份名称"""
    if pd.isna(province):
        return province
    if isinstance(province, (int, float)):
        code_str = str(int(province))
    else:
        code_str = str(province).strip()

    if code_str in PROVINCE_CODE_TO_NAME:
        return PROVINCE_CODE_TO_NAME[code_str]
    elif code_str in PROVINCE_TO_REGION:
        return code_str
    return code_str


def load_results(year):
    """加载分析结果"""
    year_input_dir = os.path.join(INPUT_DIR, str(year))
    stats_file = os.path.join(year_input_dir, "province_stats.csv")
    occupation_file = os.path.join(year_input_dir, "occupation_bias.csv")
    domestic_work_file = os.path.join(year_input_dir, "domestic_work_bias.csv")

    if not os.path.exists(stats_file) or not os.path.exists(occupation_file):
        print(f"❌ 未找到 {year} 年的分析结果")
        return None, None, None

    stats_df = pd.read_csv(stats_file)
    occupation_df = pd.read_csv(occupation_file)

    # 加载家务分工数据
    domestic_work_df = None
    if os.path.exists(domestic_work_file):
        domestic_work_df = pd.read_csv(domestic_work_file)
        print(f"✓ 加载了家务分工数据")

    # 只对2020年数据转换省份编码，2024及其他年份不需要
    if year == 2020:
        print(f"  检测到2020年数据，将转换省份编码...")
        for df in [stats_df, occupation_df]:
            df["province"] = df["province"].apply(convert_province_code)
            df["region"] = df["province"].map(PROVINCE_TO_REGION)
            df["region"] = df["region"].fillna("未知区域")

        if domestic_work_df is not None:
            domestic_work_df["province"] = domestic_work_df["province"].apply(
                convert_province_code
            )
            domestic_work_df["region"] = domestic_work_df["province"].map(
                PROVINCE_TO_REGION
            )
            domestic_work_df["region"] = domestic_work_df["region"].fillna("未知区域")

        # 过滤未知省份
        stats_df = stats_df[stats_df["province"] != "未知"].copy()
        occupation_df = occupation_df[occupation_df["province"] != "未知"].copy()
        if domestic_work_df is not None:
            domestic_work_df = domestic_work_df[
                domestic_work_df["province"] != "未知"
            ].copy()
    else:
        print(f"  {year}年数据使用省份名称，无需转换编码")
        # 直接添加区域信息
        for df in [stats_df, occupation_df]:
            df["region"] = df["province"].map(PROVINCE_TO_REGION)
            df["region"] = df["region"].fillna("未知区域")

        if domestic_work_df is not None:
            domestic_work_df["region"] = domestic_work_df["province"].map(
                PROVINCE_TO_REGION
            )
            domestic_work_df["region"] = domestic_work_df["region"].fillna("未知区域")

    print(f"✓ 加载了 {len(stats_df)} 个省份的数据")
    print(f"✓ 加载了 {len(occupation_df)} 条职业-省份记录")
    if domestic_work_df is not None:
        print(f"✓ 加载了 {len(domestic_work_df)} 条家务分工记录")

    return stats_df, occupation_df, domestic_work_df


def load_china_map(shapefile_path=None):
    """加载中国地图shapefile"""
    if shapefile_path is None:
        shapefile_dir = "configs/china_shp"
        if os.path.exists(shapefile_dir) and os.path.isdir(shapefile_dir):
            shp_files = glob.glob(os.path.join(shapefile_dir, "*.shp"))
            if shp_files:
                shapefile_path = shp_files[0]
            else:
                return None
        else:
            return None

    if os.path.isdir(shapefile_path):
        shp_files = glob.glob(os.path.join(shapefile_path, "*.shp"))
        if shp_files:
            shapefile_path = shp_files[0]
        else:
            return None

    if not os.path.exists(shapefile_path):
        return None

    try:
        gdf = gpd.read_file(shapefile_path)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326", allow_override=True)
        return gdf
    except Exception as e:
        print(f"❌ 加载地图文件失败: {e}")
        return None


# ============================================================================
# 1. 职业与性别分析
# ============================================================================


def visualize_occupation_gender(
    stats_df,
    occupation_df,
    year,
    method="similarity",  # 'similarity' or 'projection'
    shapefile_path=None,
):
    """
    职业与性别关系可视化

    Args:
        stats_df: 省份统计数据
        occupation_df: 职业数据
        year: 年份
        method: 'similarity' (余弦相似度差值) 或 'projection' (性别轴投影)
        shapefile_path: 地图文件路径
    """
    print(f"\n{'='*70}")
    print(f"📊 职业与性别分析 - {method.upper()} 方法")
    print(f"{'='*70}\n")

    # 确定使用的列名
    if method == "similarity":
        score_col = "bias_score"
        std_col = "occupation_std_bias"
        mean_col = "occupation_mean_bias"
        method_label = "余弦相似度差值"
        method_suffix = "similarity"
    else:  # projection
        score_col = "projection_score"
        std_col = "occupation_std_projection"
        mean_col = "occupation_mean_projection"
        method_label = "性别轴投影"
        method_suffix = "projection"

    # (1) 省份地图 - 性别隔离程度
    print(f"  (1) 生成省份性别隔离程度地图...")
    plot_occupation_province_map(
        stats_df, year, method_suffix, std_col, method_label, shapefile_path
    )

    # (2) 省份横向条形图
    print(f"  (2) 生成省份性别隔离排名图...")
    plot_occupation_province_ranking(
        stats_df, year, method_suffix, std_col, method_label
    )

    # (3) Top 5 省份及其 Top 5 女性/男性职业
    print(f"  (3) 生成Top 5省份的职业分析...")
    plot_top_provinces_occupations(
        stats_df, occupation_df, year, method_suffix, score_col, std_col, method_label
    )

    # (4) 职业列表 - 最偏女性/男性
    print(f"  (4) 生成职业性别偏向排名...")
    plot_occupation_ranking(occupation_df, year, method_suffix, score_col, method_label)

    # (5) 职业案例研究
    print(f"  (5) 生成职业案例研究...")
    case_study_occupations = [
        "护士",
        "幼师",
        "保姆",
        "程序员",
        "工程师",
        "保安",
        "产品经理",
        "UP主",
        "滴滴司机",
    ]
    for occ in case_study_occupations:
        if occ in occupation_df["occupation"].values:
            plot_occupation_case_study(
                occupation_df, occ, year, method_suffix, score_col, method_label
            )


def plot_occupation_province_map(
    stats_df, year, method_suffix, std_col, method_label, shapefile_path=None
):
    """(1) 省份地图 - 性别隔离程度"""
    china_map = load_china_map(shapefile_path)

    if china_map is None:
        print("    ⚠️  无法加载地图，跳过地图绘制")
        return

    # 识别省份名称列
    possible_name_cols = [
        "ADM1_ZH",
        "admin1",
        "NAME_1",
        "name_1",
        "NAME",
        "name",
        "PROV",
        "prov",
        "Province",
        "province",
        "NAME_CH",
        "name_ch",
        "FCNAME",
        "fcname",
    ]
    name_col = None
    for col in possible_name_cols:
        if col in china_map.columns:
            name_col = col
            break

    if name_col is None:
        print("    ⚠️  无法识别省份名称列")
        return

    # 标准化省份名称并合并
    stats_df_copy = stats_df.copy()
    stats_df_copy["province_full"] = stats_df_copy["province"].map(
        PROVINCE_NAME_MAPPING
    )
    stats_df_copy["province_full"] = stats_df_copy["province_full"].fillna(
        stats_df_copy["province"]
    )

    china_map_merged = china_map.merge(
        stats_df_copy, left_on=name_col, right_on="province_full", how="left"
    )

    # 绘制地图
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    colors = [
        "#fff5f0",
        "#fee5d9",
        "#fcbba1",
        "#fc9272",
        "#fb6a4a",
        "#ef3b2c",
        "#cb181d",
        "#99000d",
    ]
    cmap = LinearSegmentedColormap.from_list("segregation", colors)

    china_map_merged.plot(
        column=std_col,
        cmap=cmap,
        linewidth=0.5,
        edgecolor="white",
        legend=True,
        ax=ax,
        missing_kwds={"color": "lightgrey", "label": "无数据"},
        legend_kwds={
            "label": f"性别隔离指数（{method_label}）",
            "orientation": "vertical",
            "shrink": 0.6,
        },
    )

    # 添加省份标签
    for idx, row in china_map_merged.iterrows():
        if pd.notna(row.get(std_col)):
            centroid = row["geometry"].centroid
            ax.annotate(
                text=f"{row['province']}\n{row[std_col]:.3f}",
                xy=(centroid.x, centroid.y),
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    alpha=0.7,
                    edgecolor="none",
                ),
            )

    ax.set_title(
        f"中国各省份职业性别隔离程度地图 ({year}年 - {method_label})\n"
        + "颜色越深 = 性别隔离程度越高",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.axis("off")

    plt.tight_layout()
    output_file = generate_output_path(f"occupation_province_map_{method_suffix}", year)
    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    print(f"    ✓ 保存: {output_file}")
    plt.close()


def plot_occupation_province_ranking(
    stats_df, year, method_suffix, std_col, method_label
):
    """(2) 省份横向条形图 - 性别隔离排名"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    stats_sorted = stats_df.sort_values(std_col, ascending=True)

    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(stats_sorted)))
    bars = ax.barh(stats_sorted["province"], stats_sorted[std_col], color=colors)

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}",
            va="center",
            fontsize=9,
        )

    ax.set_xlabel(f"性别隔离指数（{method_label}）", fontsize=12, fontweight="bold")
    ax.set_title(
        f"各省份职业性别隔离程度排名 ({year}年 - {method_label})",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    plt.tight_layout()
    output_file = generate_output_path(
        f"occupation_province_ranking_{method_suffix}", year
    )
    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    print(f"    ✓ 保存: {output_file}")
    plt.close()


def plot_top_provinces_occupations(
    stats_df, occupation_df, year, method_suffix, score_col, std_col, method_label
):
    """(3) Top 5 省份及其 Top 5 女性/男性职业"""
    top_provinces = stats_df.nlargest(5, std_col)["province"].tolist()

    for province in top_provinces:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        prov_df = occupation_df[occupation_df["province"] == province]

        # Top 5 女性职业
        top_female = prov_df.nlargest(5, score_col).sort_values(score_col)
        bars1 = ax1.barh(
            top_female["occupation"],
            top_female[score_col],
            color="#2ca02c",
            alpha=0.7,
        )
        for bar in bars1:
            width = bar.get_width()
            ax1.text(
                width + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{width:+.3f}",
                va="center",
                fontsize=9,
            )
        ax1.set_xlabel(
            f"性别偏向分数（{method_label}）", fontsize=11, fontweight="bold"
        )
        ax1.set_title(f"Top 5 偏女性职业", fontsize=12, fontweight="bold")
        ax1.grid(axis="x", alpha=0.3)

        # Top 5 男性职业
        top_male = prov_df.nsmallest(5, score_col).sort_values(score_col)
        bars2 = ax2.barh(
            top_male["occupation"],
            top_male[score_col],
            color="#d62728",
            alpha=0.7,
        )
        for bar in bars2:
            width = bar.get_width()
            ax2.text(
                width - 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{width:+.3f}",
                va="center",
                ha="right",
                fontsize=9,
            )
        ax2.set_xlabel(
            f"性别偏向分数（{method_label}）", fontsize=11, fontweight="bold"
        )
        ax2.set_title(f"Top 5 偏男性职业", fontsize=12, fontweight="bold")
        ax2.grid(axis="x", alpha=0.3)

        fig.suptitle(
            f"{province} - 职业性别偏向 ({year}年 - {method_label})",
            fontsize=14,
            fontweight="bold",
        )

        plt.tight_layout()
        output_file = generate_output_path(
            f"occupation_top_province_{province}_{method_suffix}", year
        )
        plt.savefig(output_file, format="pdf", bbox_inches="tight")
        print(f"    ✓ 保存: {output_file}")
        plt.close()


def plot_occupation_ranking(
    occupation_df, year, method_suffix, score_col, method_label
):
    """(4) 职业列表 - 最偏女性/男性"""
    occ_avg = (
        occupation_df.groupby("occupation")[score_col]
        .mean()
        .sort_values(ascending=True)
    )

    # 取Top 10 女性和Top 10 男性
    top_male = occ_avg.head(10)
    top_female = occ_avg.tail(10)
    combined = pd.concat([top_male, top_female])

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    colors = ["#d62728" if x < 0 else "#2ca02c" for x in combined.values]
    bars = ax.barh(combined.index, combined.values, color=colors, alpha=0.7)

    ax.axvline(x=0, color="black", linestyle="--", linewidth=1)

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + (0.005 if width > 0 else -0.005),
            bar.get_y() + bar.get_height() / 2,
            f"{width:+.3f}",
            va="center",
            ha="left" if width > 0 else "right",
            fontsize=9,
        )

    ax.set_xlabel(f"平均性别偏向分数（{method_label}）", fontsize=12, fontweight="bold")
    ax.set_title(
        f"职业性别偏向排名 ({year}年 - {method_label})\n" + "负值=偏男性，正值=偏女性",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    output_file = generate_output_path(f"occupation_ranking_{method_suffix}", year)
    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    print(f"    ✓ 保存: {output_file}")
    plt.close()


def plot_occupation_case_study(
    occupation_df, occ_name, year, method_suffix, score_col, method_label
):
    """(5) 职业案例研究 - 各省份偏向分数"""
    occ_df = occupation_df[occupation_df["occupation"] == occ_name].copy()

    if len(occ_df) == 0:
        return

    occ_df = occ_df.sort_values(score_col, ascending=True)

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    colors = ["#d62728" if x < 0 else "#2ca02c" for x in occ_df[score_col]]
    bars = ax.barh(occ_df["province"], occ_df[score_col], color=colors, alpha=0.7)

    ax.axvline(x=0, color="black", linestyle="--", linewidth=1)

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + (0.005 if width > 0 else -0.005),
            bar.get_y() + bar.get_height() / 2,
            f"{width:+.3f}",
            va="center",
            ha="left" if width > 0 else "right",
            fontsize=8,
        )

    ax.set_xlabel(f"性别偏向分数（{method_label}）", fontsize=12, fontweight="bold")
    ax.set_title(
        f'"{occ_name}" 在各省份的性别偏向 ({year}年 - {method_label})\n'
        + "负值=偏男性，正值=偏女性",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    output_file = generate_output_path(
        f"occupation_case_{occ_name}_{method_suffix}", year
    )
    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    print(f"    ✓ 保存: {output_file}")
    plt.close()


# ============================================================================
# 2. 家务劳动与性别分析
# ============================================================================


def visualize_domestic_work_gender(
    stats_df,
    domestic_work_df,
    year,
    method="similarity",  # 'similarity' or 'projection'
    shapefile_path=None,
):
    """
    家务劳动与性别关系可视化

    Args:
        stats_df: 省份统计数据
        domestic_work_df: 家务分工数据
        year: 年份
        method: 'similarity' (余弦相似度差值) 或 'projection' (性别轴投影)
        shapefile_path: 地图文件路径
    """
    if domestic_work_df is None or len(domestic_work_df) == 0:
        print("⚠️  没有家务分工数据，跳过家务劳动分析")
        return

    print(f"\n{'='*70}")
    print(f"🏠 家务劳动与性别分析 - {method.upper()} 方法")
    print(f"{'='*70}\n")

    # 确定使用的列名
    if method == "similarity":
        score_col = "bias_score"
        gap_col = "domain_bias_gap"
        work_mean_col = "work_mean_bias"
        family_mean_col = "family_mean_bias"
        method_label = "余弦相似度差值"
        method_suffix = "similarity"
    else:  # projection
        score_col = "projection_score"
        gap_col = "domain_projection_gap"
        work_mean_col = "work_mean_projection"
        family_mean_col = "family_mean_projection"
        method_label = "性别轴投影"
        method_suffix = "projection"

    # 计算每个省份的 work 和 family 平均分数
    province_stats = compute_province_domestic_stats(
        domestic_work_df, score_col, gap_col
    )

    # (1) 省份地图
    print(f"  (1) 生成省份家务分工性别差异地图...")
    plot_domestic_work_province_map(
        province_stats, year, method_suffix, method_label, shapefile_path
    )

    # (2) 省份横向条形图
    print(f"  (2) 生成省份家务分工性别差异排名...")
    plot_domestic_work_province_ranking(
        province_stats, year, method_suffix, method_label
    )

    # (3) Top 5 省份的详细分析
    print(f"  (3) 生成Top 5省份的家务分工分析...")
    plot_top_provinces_domestic_work(
        province_stats, domestic_work_df, year, method_suffix, score_col, method_label
    )

    # (4) Top 5 domestic/work 词汇案例研究
    print(f"  (4) 生成家务/工作词汇案例研究...")
    plot_domestic_work_case_studies(
        domestic_work_df, year, method_suffix, score_col, method_label
    )


def compute_province_domestic_stats(domestic_work_df, score_col, gap_col):
    """计算每个省份的 work/family 统计数据"""
    province_stats = []

    for province in domestic_work_df["province"].unique():
        prov_df = domestic_work_df[domestic_work_df["province"] == province]

        work_df = prov_df[prov_df["word_type"] == "work"]
        family_df = prov_df[prov_df["word_type"] == "family"]

        if len(work_df) > 0 and len(family_df) > 0:
            work_mean = work_df[score_col].mean()
            family_mean = family_df[score_col].mean()
            gap = family_mean - work_mean  # 正值表示 family 比 work 更偏女性

            province_stats.append(
                {
                    "province": province,
                    "work_mean": work_mean,
                    "family_mean": family_mean,
                    "gap": gap,
                }
            )

    return pd.DataFrame(province_stats)


def plot_domestic_work_province_map(
    province_stats, year, method_suffix, method_label, shapefile_path=None
):
    """(1) 省份地图 - 家务分工性别差异"""
    china_map = load_china_map(shapefile_path)

    if china_map is None:
        print("    ⚠️  无法加载地图，跳过地图绘制")
        return

    # 识别省份名称列
    possible_name_cols = [
        "ADM1_ZH",
        "admin1",
        "NAME_1",
        "name_1",
        "NAME",
        "name",
        "PROV",
        "prov",
        "Province",
        "province",
        "NAME_CH",
        "name_ch",
        "FCNAME",
        "fcname",
    ]
    name_col = None
    for col in possible_name_cols:
        if col in china_map.columns:
            name_col = col
            break

    if name_col is None:
        print("    ⚠️  无法识别省份名称列")
        return

    # 标准化省份名称并合并
    province_stats_copy = province_stats.copy()
    province_stats_copy["province_full"] = province_stats_copy["province"].map(
        PROVINCE_NAME_MAPPING
    )
    province_stats_copy["province_full"] = province_stats_copy["province_full"].fillna(
        province_stats_copy["province"]
    )

    china_map_merged = china_map.merge(
        province_stats_copy, left_on=name_col, right_on="province_full", how="left"
    )

    # 绘制地图
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    # 使用红绿配色：负值（work更偏女性）为绿色，正值（family更偏女性）为红色
    cmap = plt.cm.RdYlGn_r

    china_map_merged.plot(
        column="gap",
        cmap=cmap,
        linewidth=0.5,
        edgecolor="white",
        legend=True,
        ax=ax,
        missing_kwds={"color": "lightgrey", "label": "无数据"},
        legend_kwds={
            "label": f"家务分工性别差异（{method_label}）\n正=家庭偏女性",
            "orientation": "vertical",
            "shrink": 0.6,
        },
    )

    # 添加省份标签
    for idx, row in china_map_merged.iterrows():
        if pd.notna(row.get("gap")):
            centroid = row["geometry"].centroid
            ax.annotate(
                text=f"{row['province']}\n{row['gap']:+.3f}",
                xy=(centroid.x, centroid.y),
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    alpha=0.7,
                    edgecolor="none",
                ),
            )

    ax.set_title(
        f"中国各省份家务分工性别差异地图 ({year}年 - {method_label})\n"
        + "正值=家庭场域更偏女性，负值=工作场域更偏女性",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.axis("off")

    plt.tight_layout()
    output_file = generate_output_path(
        f"domestic_work_province_map_{method_suffix}", year
    )
    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    print(f"    ✓ 保存: {output_file}")
    plt.close()


def plot_domestic_work_province_ranking(
    province_stats, year, method_suffix, method_label
):
    """(2) 省份横向条形图 - 家务分工性别差异排名"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    stats_sorted = province_stats.sort_values("gap", ascending=True)

    colors = ["#2ca02c" if x < 0 else "#d62728" for x in stats_sorted["gap"]]
    bars = ax.barh(
        stats_sorted["province"], stats_sorted["gap"], color=colors, alpha=0.7
    )

    ax.axvline(x=0, color="black", linestyle="--", linewidth=1)

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + (0.005 if width > 0 else -0.005),
            bar.get_y() + bar.get_height() / 2,
            f"{width:+.3f}",
            va="center",
            ha="left" if width > 0 else "right",
            fontsize=9,
        )

    ax.set_xlabel(
        f"家务分工性别差异（{method_label}）\n负=工作偏女性，正=家庭偏女性",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_title(
        f"各省份家务分工性别差异排名 ({year}年 - {method_label})",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    output_file = generate_output_path(
        f"domestic_work_province_ranking_{method_suffix}", year
    )
    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    print(f"    ✓ 保存: {output_file}")
    plt.close()


def plot_top_provinces_domestic_work(
    province_stats, domestic_work_df, year, method_suffix, score_col, method_label
):
    """(3) Top 5 省份的家务分工详细分析"""
    # 取gap绝对值最大的5个省份
    province_stats["abs_gap"] = province_stats["gap"].abs()
    top_provinces = province_stats.nlargest(5, "abs_gap")["province"].tolist()

    for province in top_provinces:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        prov_df = domestic_work_df[domestic_work_df["province"] == province]
        work_df = prov_df[prov_df["word_type"] == "work"]
        family_df = prov_df[prov_df["word_type"] == "family"]

        # 1. Work vs Family 平均分数对比
        prov_stats = province_stats[province_stats["province"] == province].iloc[0]
        x = [0, 1]
        heights = [prov_stats["work_mean"], prov_stats["family_mean"]]
        colors_bar = ["#1f77b4", "#ff7f0e"]
        bars = ax1.bar(x, heights, color=colors_bar, alpha=0.7, width=0.6)
        ax1.set_xticks(x)
        ax1.set_xticklabels(["工作场域", "家庭场域"])
        ax1.set_ylabel(
            f"平均性别偏向（{method_label}）", fontsize=11, fontweight="bold"
        )
        ax1.set_title(f"{province} - 场域性别偏向对比", fontsize=12, fontweight="bold")
        ax1.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax1.grid(axis="y", alpha=0.3)

        for bar, h in zip(bars, heights):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                h + (0.01 if h > 0 else -0.01),
                f"{h:+.3f}",
                ha="center",
                va="bottom" if h > 0 else "top",
                fontsize=10,
                fontweight="bold",
            )

        # 2. Top 5 Family 词（最偏女性）
        if len(family_df) > 0:
            top_family = family_df.nlargest(5, score_col).sort_values(score_col)
            bars2 = ax2.barh(
                top_family["word"],
                top_family[score_col],
                color="#ff7f0e",
                alpha=0.7,
            )
            ax2.axvline(x=0, color="black", linestyle="--", linewidth=1)
            for bar in bars2:
                width = bar.get_width()
                ax2.text(
                    width + 0.005,
                    bar.get_y() + bar.get_height() / 2,
                    f"{width:+.3f}",
                    va="center",
                    fontsize=9,
                )
            ax2.set_xlabel(
                f"性别偏向（{method_label}）", fontsize=11, fontweight="bold"
            )
            ax2.set_title("Top 5 家庭场域词（偏女性）", fontsize=12, fontweight="bold")
            ax2.grid(axis="x", alpha=0.3)
        else:
            ax2.text(
                0.5, 0.5, "无数据", ha="center", va="center", transform=ax2.transAxes
            )

        # 3. Top 5 Work 词（最偏女性）
        if len(work_df) > 0:
            top_work_female = work_df.nlargest(5, score_col).sort_values(score_col)
            bars3 = ax3.barh(
                top_work_female["word"],
                top_work_female[score_col],
                color="#1f77b4",
                alpha=0.7,
            )
            ax3.axvline(x=0, color="black", linestyle="--", linewidth=1)
            for bar in bars3:
                width = bar.get_width()
                ax3.text(
                    width + 0.005,
                    bar.get_y() + bar.get_height() / 2,
                    f"{width:+.3f}",
                    va="center",
                    fontsize=9,
                )
            ax3.set_xlabel(
                f"性别偏向（{method_label}）", fontsize=11, fontweight="bold"
            )
            ax3.set_title("Top 5 工作场域词（偏女性）", fontsize=12, fontweight="bold")
            ax3.grid(axis="x", alpha=0.3)
        else:
            ax3.text(
                0.5, 0.5, "无数据", ha="center", va="center", transform=ax3.transAxes
            )

        # 4. Top 5 Work 词（最偏男性）
        if len(work_df) > 0:
            top_work_male = work_df.nsmallest(5, score_col).sort_values(score_col)
            bars4 = ax4.barh(
                top_work_male["word"],
                top_work_male[score_col],
                color="#d62728",
                alpha=0.7,
            )
            ax4.axvline(x=0, color="black", linestyle="--", linewidth=1)
            for bar in bars4:
                width = bar.get_width()
                ax4.text(
                    width - 0.005,
                    bar.get_y() + bar.get_height() / 2,
                    f"{width:+.3f}",
                    va="center",
                    ha="right",
                    fontsize=9,
                )
            ax4.set_xlabel(
                f"性别偏向（{method_label}）", fontsize=11, fontweight="bold"
            )
            ax4.set_title("Top 5 工作场域词（偏男性）", fontsize=12, fontweight="bold")
            ax4.grid(axis="x", alpha=0.3)
        else:
            ax4.text(
                0.5, 0.5, "无数据", ha="center", va="center", transform=ax4.transAxes
            )

        fig.suptitle(
            f"{province} - 家务分工性别分析 ({year}年 - {method_label})",
            fontsize=14,
            fontweight="bold",
        )

        plt.tight_layout()
        output_file = generate_output_path(
            f"domestic_work_top_province_{province}_{method_suffix}", year
        )
        plt.savefig(output_file, format="pdf", bbox_inches="tight")
        print(f"    ✓ 保存: {output_file}")
        plt.close()


def plot_domestic_work_case_studies(
    domestic_work_df, year, method_suffix, score_col, method_label
):
    """(4) 家务/工作词汇案例研究"""
    # 计算每个词的平均分数
    word_avg = domestic_work_df.groupby("word")[score_col].mean().sort_values()

    # Top 5 family 词（最偏女性）
    family_words = domestic_work_df[domestic_work_df["word_type"] == "family"][
        "word"
    ].unique()
    family_word_avg = word_avg[word_avg.index.isin(family_words)]
    if len(family_word_avg) > 0:
        top_family_words = family_word_avg.nlargest(5).index.tolist()

        for word in top_family_words:
            plot_domestic_word_case_study(
                domestic_work_df,
                word,
                year,
                method_suffix,
                score_col,
                method_label,
                "family",
            )

    # Top 5 work 词（最偏女性 或 最偏男性）
    work_words = domestic_work_df[domestic_work_df["word_type"] == "work"][
        "word"
    ].unique()
    work_word_avg = word_avg[word_avg.index.isin(work_words)]
    if len(work_word_avg) > 0:
        # 取最偏女性的5个
        top_work_female = work_word_avg.nlargest(5).index.tolist()
        for word in top_work_female:
            plot_domestic_word_case_study(
                domestic_work_df,
                word,
                year,
                method_suffix,
                score_col,
                method_label,
                "work",
            )

        # 取最偏男性的5个
        top_work_male = work_word_avg.nsmallest(5).index.tolist()
        for word in top_work_male:
            plot_domestic_word_case_study(
                domestic_work_df,
                word,
                year,
                method_suffix,
                score_col,
                method_label,
                "work",
            )


def plot_domestic_word_case_study(
    domestic_work_df, word, year, method_suffix, score_col, method_label, word_type
):
    """单个家务/工作词汇的案例研究"""
    word_df = domestic_work_df[domestic_work_df["word"] == word].copy()

    if len(word_df) == 0:
        return

    word_df = word_df.sort_values(score_col, ascending=True)

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    colors = ["#d62728" if x < 0 else "#2ca02c" for x in word_df[score_col]]
    bars = ax.barh(word_df["province"], word_df[score_col], color=colors, alpha=0.7)

    ax.axvline(x=0, color="black", linestyle="--", linewidth=1)

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + (0.005 if width > 0 else -0.005),
            bar.get_y() + bar.get_height() / 2,
            f"{width:+.3f}",
            va="center",
            ha="left" if width > 0 else "right",
            fontsize=8,
        )

    word_type_label = "家庭场域" if word_type == "family" else "工作场域"

    ax.set_xlabel(f"性别偏向分数（{method_label}）", fontsize=12, fontweight="bold")
    ax.set_title(
        f'"{word}" ({word_type_label}) 在各省份的性别偏向 ({year}年 - {method_label})\n'
        + "负值=偏男性，正值=偏女性",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    output_file = generate_output_path(
        f"domestic_word_case_{word}_{method_suffix}", year
    )
    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    print(f"    ✓ 保存: {output_file}")
    plt.close()


# ============================================================================
# 主函数
# ============================================================================


def main(year: int, shapefile: str = None):
    """
    运行可视化分析

    Args:
        year: 年份
        shapefile: 中国地图shapefile路径（可选）
    """
    print(f"\n{'='*70}")
    print(f"🎨 开始生成 {year} 年省份性别偏向可视化（重构版）")
    print(f"{'='*70}\n")

    # 加载数据
    stats_df, occupation_df, domestic_work_df = load_results(year)
    if stats_df is None or occupation_df is None:
        return

    # ========== 1. 职业与性别分析 ==========

    # 1a. 基于相似度方法
    visualize_occupation_gender(
        stats_df, occupation_df, year, method="similarity", shapefile_path=shapefile
    )

    # 1b. 基于投影方法
    visualize_occupation_gender(
        stats_df, occupation_df, year, method="projection", shapefile_path=shapefile
    )

    # ========== 2. 家务劳动与性别分析 ==========

    if domestic_work_df is not None:
        # 2a. 基于相似度方法
        visualize_domestic_work_gender(
            stats_df,
            domestic_work_df,
            year,
            method="similarity",
            shapefile_path=shapefile,
        )

        # 2b. 基于投影方法
        visualize_domestic_work_gender(
            stats_df,
            domestic_work_df,
            year,
            method="projection",
            shapefile_path=shapefile,
        )

    print(f"\n{'='*70}")
    print(f"✅ 可视化完成！所有文件已保存到: {OUTPUT_DIR}/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
