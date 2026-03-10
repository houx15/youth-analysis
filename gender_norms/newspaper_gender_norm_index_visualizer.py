"""
Gender Norm Index 可视化器 (报纸数据版)

基于报纸语料库的性别规范指数可视化:
1. 跨省可比性诊断图: 各省所有概念词投影值的boxplot
2. 三个index的分布图: 各省效应量的histogram或density plot
3. 地图可视化: 三个index的省级choropleth map

输入: newspaper_data/gender_norm_index/ 下的分析结果
输出: newspaper_data/gender_norm_index/figures/ 下的PDF图表

数据年份: 2019年报纸语料
"""

import os
import glob
import warnings
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns

warnings.filterwarnings("ignore")

try:
    import geopandas as gpd

    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    print("[警告] geopandas未安装, 将跳过地图可视化")

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


# =============================================================================
# 配置
# =============================================================================

INPUT_DIR = "gender_norms/newspaper_data/gender_norm_index"
SHAPEFILE_DIR = "configs/china_shp"

DATA_YEAR = 2019
DATA_SOURCE = "News"

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

CN_TO_EN_PROVINCE = {
    "北京": "Beijing",
    "天津": "Tianjin",
    "上海": "Shanghai",
    "重庆": "Chongqing",
    "河北": "Hebei",
    "山西": "Shanxi",
    "辽宁": "Liaoning",
    "吉林": "Jilin",
    "黑龙江": "Heilongjiang",
    "江苏": "Jiangsu",
    "浙江": "Zhejiang",
    "安徽": "Anhui",
    "福建": "Fujian",
    "江西": "Jiangxi",
    "山东": "Shandong",
    "河南": "Henan",
    "湖北": "Hubei",
    "湖南": "Hunan",
    "广东": "Guangdong",
    "海南": "Hainan",
    "四川": "Sichuan",
    "贵州": "Guizhou",
    "云南": "Yunnan",
    "陕西": "Shaanxi",
    "甘肃": "Gansu",
    "青海": "Qinghai",
    "广西": "Guangxi",
    "内蒙古": "Inner Mongolia",
    "宁夏": "Ningxia",
    "新疆": "Xinjiang",
    "西藏": "Tibet",
    "中国香港": "Hong Kong",
    "中国澳门": "Macau",
    "中国台湾": "Taiwan",
}


# =============================================================================
# 数据加载
# =============================================================================


def load_results():
    """加载分析结果"""
    files = {
        "projections": os.path.join(INPUT_DIR, "word_projections.csv"),
        "weat": os.path.join(INPUT_DIR, "weat_results.csv"),
        "index": os.path.join(INPUT_DIR, "gender_norm_index.csv"),
        "stats": os.path.join(INPUT_DIR, "province_projection_stats.csv"),
    }

    data = {}
    for name, path in files.items():
        if os.path.exists(path):
            data[name] = pd.read_csv(path)
            print(f"  [加载] {name}: {len(data[name])} 条记录")
        else:
            print(f"  [警告] 文件不存在: {path}")
            data[name] = None

    return data


def load_china_map(shapefile_path: Optional[str] = None):
    """加载中国地图shapefile"""
    if not HAS_GEOPANDAS:
        return None

    if shapefile_path is None:
        if os.path.exists(SHAPEFILE_DIR) and os.path.isdir(SHAPEFILE_DIR):
            shp_files = glob.glob(os.path.join(SHAPEFILE_DIR, "*.shp"))
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
        print(f"  [错误] 加载地图文件失败: {e}")
        return None


def generate_output_path(filename: str) -> str:
    """生成输出文件路径"""
    figures_dir = os.path.join(INPUT_DIR, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    date_prefix = datetime.now().strftime("%Y%m%d")
    return os.path.join(figures_dir, f"{date_prefix}_{filename}.pdf")


# =============================================================================
# 可视化函数
# =============================================================================


def plot_projection_boxplot(projection_df: pd.DataFrame):
    """
    1. 跨省可比性诊断图: 各省所有概念词投影值的boxplot
    """
    print(f"\n  生成跨省可比性诊断图...")

    if projection_df is None or len(projection_df) == 0:
        print(f"    [跳过] 无投影数据")
        return

    projection_df["province_en"] = projection_df["province"].map(CN_TO_EN_PROVINCE)
    province_order = (
        projection_df.groupby("province_en")["cosine_sim"]
        .median()
        .sort_values()
        .index.tolist()
    )

    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    ax1 = axes[0]
    sns.boxplot(
        data=projection_df,
        x="province_en",
        y="cosine_sim",
        order=province_order,
        ax=ax1,
        palette="Set3",
    )
    ax1.axhline(y=0, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax1.set_xlabel("Province", fontsize=12, fontweight="bold")
    ax1.set_ylabel(
        "Cosine Similarity (Gender Axis Projection)", fontsize=12, fontweight="bold"
    )
    ax1.set_title(
        f"Distribution of Concept Words on Gender Axis ({DATA_SOURCE}, Year: {DATA_YEAR})\n"
        "Positive=Female-Biased, Negative=Male-Biased",
        fontsize=14,
        fontweight="bold",
    )
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    category_order = [
        "family",
        "work",
        "leadership",
        "non_leadership",
        "stem",
        "non_stem",
    ]
    category_labels = {
        "family": "Family",
        "work": "Work",
        "leadership": "Leadership",
        "non_leadership": "Non-Leadership",
        "stem": "STEM",
        "non_stem": "Non-STEM",
    }

    existing_categories = [
        c for c in category_order if c in projection_df["category"].unique()
    ]

    sns.violinplot(
        data=projection_df[projection_df["category"].isin(existing_categories)],
        x="category",
        y="cosine_sim",
        order=existing_categories,
        ax=ax2,
        palette="husl",
        inner="box",
    )
    ax2.axhline(y=0, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax2.set_xlabel("Category", fontsize=12, fontweight="bold")
    ax2.set_ylabel(
        "Cosine Similarity (Gender Axis Projection)", fontsize=12, fontweight="bold"
    )
    ax2.set_title(
        f"Distribution of Concept Words on Gender Axis ({DATA_SOURCE}, Year: {DATA_YEAR})\nPositive=Female-Biased, Negative=Male-Biased",
        fontsize=14,
        fontweight="bold",
    )
    ax2.set_xticklabels([category_labels.get(c, c) for c in existing_categories])
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_file = generate_output_path("comparability_diagnostic")
    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    print(f"    [保存] {output_file}")
    plt.close()


def plot_weat_distribution(weat_df: pd.DataFrame):
    """
    2. 三个index的分布图: 各省效应量的histogram或density plot
    """
    print(f"\n  生成WEAT效应量分布图...")

    if weat_df is None or len(weat_df) == 0:
        print(f"    [跳过] 无WEAT数据")
        return

    dimensions = ["work_family", "leadership", "stem"]
    dim_labels = {
        "work_family": "Work-Family\n(Positive=Women Associated with Family)",
        "leadership": "Leadership\n(Positive=Men Associated with Leadership)",
        "stem": "STEM\n(Positive=Men Associated with STEM)",
    }
    colors = ["#2ecc71", "#e74c3c", "#3498db"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, dim in enumerate(dimensions):
        ax = axes[i]
        dim_data = weat_df[weat_df["dimension"] == dim]["cohens_d"]

        if len(dim_data) == 0:
            ax.text(
                0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes
            )
            continue

        sns.histplot(
            dim_data,
            kde=True,
            ax=ax,
            color=colors[i],
            alpha=0.6,
            edgecolor="white",
        )

        mean_val = dim_data.mean()
        median_val = dim_data.median()
        ax.axvline(
            x=mean_val,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_val:.3f}",
        )
        ax.axvline(
            x=median_val,
            color="blue",
            linestyle=":",
            linewidth=2,
            label=f"Median: {median_val:.3f}",
        )
        ax.axvline(x=0, color="black", linestyle="-", linewidth=1, alpha=0.5)

        ax.set_xlabel("Cohen's d", fontsize=12, fontweight="bold")
        ax.set_ylabel("Frequency", fontsize=12, fontweight="bold")
        ax.set_title(dim_labels[dim], fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle(
        f"Gender Norm Index Distribution ({DATA_SOURCE}, Year: {DATA_YEAR})\n|d|≈0.2 Small Effect, |d|≈0.5 Medium Effect, |d|≈0.8 Large Effect",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    output_file = generate_output_path("weat_distribution")
    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    print(f"    [保存] {output_file}")
    plt.close()


def plot_weat_ranking(weat_df: pd.DataFrame):
    """
    各省份WEAT效应量排名条形图
    """
    print(f"\n  生成WEAT效应量排名图...")

    if weat_df is None or len(weat_df) == 0:
        print(f"    [跳过] 无WEAT数据")
        return

    dimensions = ["work_family", "leadership", "stem"]
    dim_labels = {
        "work_family": "Work-Family (d)",
        "leadership": "Leadership (d)",
        "stem": "STEM (d)",
    }
    colors = ["#2ecc71", "#e74c3c", "#3498db"]

    for dim, color in zip(dimensions, colors):
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        dim_data = weat_df[weat_df["dimension"] == dim].copy()

        dim_data["province_en"] = dim_data["province"].map(CN_TO_EN_PROVINCE)

        if len(dim_data) == 0:
            plt.close()
            continue

        dim_data = dim_data.sort_values("cohens_d", ascending=True)

        bar_colors = [color if d >= 0 else "#95a5a6" for d in dim_data["cohens_d"]]

        bars = ax.barh(
            dim_data["province_en"],
            dim_data["cohens_d"],
            color=bar_colors,
            alpha=0.7,
            edgecolor="white",
        )

        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + (0.02 if width >= 0 else -0.02),
                bar.get_y() + bar.get_height() / 2,
                f"{width:+.3f}",
                va="center",
                ha="left" if width >= 0 else "right",
                fontsize=9,
            )

        ax.axvline(x=0, color="black", linestyle="-", linewidth=1)
        ax.set_xlabel(dim_labels[dim], fontsize=12, fontweight="bold")
        ax.set_title(
            f"Ranking of {dim_labels[dim]} by Province ({DATA_SOURCE}, Year: {DATA_YEAR})",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        output_file = generate_output_path(f"weat_ranking_{dim}")
        plt.savefig(output_file, format="pdf", bbox_inches="tight")
        print(f"    [保存] {output_file}")
        plt.close()


def plot_weat_heatmap(weat_df: pd.DataFrame):
    """
    WEAT效应量热力图: 省份 × 维度
    """
    print(f"\n  生成WEAT效应量热力图...")

    if weat_df is None or len(weat_df) == 0:
        print(f"    [跳过] 无WEAT数据")
        return

    weat_df["province_en"] = weat_df["province"].map(CN_TO_EN_PROVINCE)

    pivot_df = weat_df.pivot_table(
        index="province_en",
        columns="dimension",
        values="cohens_d",
        aggfunc="first",
    )

    if "work_family" in pivot_df.columns:
        pivot_df = pivot_df.sort_values("work_family", ascending=False)

    fig, ax = plt.subplots(1, 1, figsize=(10, 12))

    sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        ax=ax,
        cbar_kws={"label": "Cohen's d"},
        linewidths=0.5,
    )

    ax.set_xlabel("Dimension", fontsize=12, fontweight="bold")
    ax.set_ylabel("Province", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Gender Norm Index Heatmap ({DATA_SOURCE}, Year: {DATA_YEAR})\nPositive=Traditional Gender Norm Direction",
        fontsize=14,
        fontweight="bold",
    )

    col_labels = {
        "work_family": "Work-Family",
        "leadership": "Leadership",
        "stem": "STEM",
    }
    ax.set_xticklabels([col_labels.get(c, c) for c in pivot_df.columns])

    plt.tight_layout()
    output_file = generate_output_path("weat_heatmap")
    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    print(f"    [保存] {output_file}")
    plt.close()


def plot_province_map(
    weat_df: pd.DataFrame,
    shapefile_path: Optional[str] = None,
):
    """
    3. 地图可视化: 三个index的省级choropleth map
    """
    print(f"\n  生成省级地图可视化...")

    if not HAS_GEOPANDAS:
        print(f"    [跳过] geopandas未安装")
        return

    if weat_df is None or len(weat_df) == 0:
        print(f"    [跳过] 无WEAT数据")
        return

    china_map = load_china_map(shapefile_path)
    if china_map is None:
        print(f"    [跳过] 无法加载地图文件")
        return

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
        print(f"    [跳过] 无法识别省份名称列")
        return

    dimensions = ["work_family", "leadership", "stem"]
    dim_labels = {
        "work_family": "Work-Family",
        "leadership": "Leadership",
        "stem": "STEM",
    }

    for dim in dimensions:
        dim_data = weat_df[weat_df["dimension"] == dim][["province", "cohens_d"]].copy()

        if len(dim_data) == 0:
            continue

        dim_data["province_full"] = dim_data["province"].map(PROVINCE_NAME_MAPPING)
        dim_data["province_full"] = dim_data["province_full"].fillna(
            dim_data["province"]
        )

        china_map_merged = china_map.merge(
            dim_data, left_on=name_col, right_on="province_full", how="left"
        )

        china_map_merged["province_en"] = china_map_merged["province"].map(
            CN_TO_EN_PROVINCE
        )

        values = dim_data["cohens_d"].dropna()
        if len(values) == 0:
            continue

        M = float(values.abs().quantile(0.95))
        if M == 0:
            M = float(values.abs().max())

        norm = TwoSlopeNorm(vcenter=0.0, vmin=-M, vmax=+M)

        fig, ax = plt.subplots(1, 1, figsize=(16, 12))

        china_map_merged.plot(
            column="cohens_d",
            cmap="RdBu_r",
            norm=norm,
            linewidth=0.5,
            edgecolor="white",
            legend=True,
            ax=ax,
            missing_kwds={"color": "lightgrey", "label": "No Data"},
            legend_kwds={
                "label": f"Cohen's d ({dim_labels[dim]})",
                "orientation": "vertical",
                "shrink": 0.6,
            },
        )

        for idx, row in china_map_merged.iterrows():
            if pd.notna(row.get("cohens_d")):
                centroid = row["geometry"].centroid
                ax.annotate(
                    text=f"{row['province_en']}\n{row['cohens_d']:+.2f}",
                    xy=(centroid.x, centroid.y),
                    ha="center",
                    va="center",
                    fontsize=7,
                    fontweight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="white",
                        alpha=0.7,
                        edgecolor="none",
                    ),
                )

        ax.set_title(
            f"Gender Norm Index: {dim_labels[dim]} ({DATA_SOURCE}, Year: {DATA_YEAR})\n"
            "Color Scale: Positive=Red, Negative=Blue, Darker=Larger |d|",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax.axis("off")

        plt.tight_layout()
        output_file = generate_output_path(f"map_{dim}")
        plt.savefig(output_file, format="pdf", bbox_inches="tight")
        print(f"    [保存] {output_file}")
        plt.close()


def plot_category_comparison(projection_df: pd.DataFrame):
    """
    分类别的省份间比较图
    """
    print(f"\n  生成类别间比较图...")

    if projection_df is None or len(projection_df) == 0:
        print(f"    [跳过] 无投影数据")
        return

    projection_df["province_en"] = projection_df["province"].map(CN_TO_EN_PROVINCE)

    category_means = (
        projection_df.groupby(["province_en", "category"])["cosine_sim"]
        .mean()
        .reset_index()
    )

    comparisons = [
        ("family", "work", "Family vs Work"),
        ("non_leadership", "leadership", "Non-Leadership vs Leadership"),
        ("non_stem", "stem", "Non-STEM vs STEM"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    for i, (cat1, cat2, title) in enumerate(comparisons):
        ax = axes[i]

        cat1_data = category_means[category_means["category"] == cat1][
            ["province_en", "cosine_sim"]
        ]
        cat2_data = category_means[category_means["category"] == cat2][
            ["province_en", "cosine_sim"]
        ]

        if len(cat1_data) == 0 or len(cat2_data) == 0:
            ax.text(
                0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes
            )
            continue

        merged = cat1_data.merge(cat2_data, on="province_en", suffixes=("_1", "_2"))

        ax.scatter(
            merged["cosine_sim_2"],
            merged["cosine_sim_1"],
            alpha=0.7,
            s=100,
            c="#3498db",
            edgecolors="white",
        )

        for _, row in merged.iterrows():
            ax.annotate(
                row["province_en"],
                (row["cosine_sim_2"], row["cosine_sim_1"]),
                fontsize=8,
                alpha=0.8,
            )

        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1)
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)

        ax.set_xlabel(f"{cat2} Average Projection", fontsize=11, fontweight="bold")
        ax.set_ylabel(f"{cat1} Average Projection", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(alpha=0.3)

    plt.suptitle(
        f"Comparison of Gender Bias between Categories ({DATA_SOURCE}, Year: {DATA_YEAR})\nPoints Above Diagonal=First Category More Female-Biased",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    output_file = generate_output_path("category_comparison")
    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    print(f"    [保存] {output_file}")
    plt.close()


# =============================================================================
# 主函数
# =============================================================================


def main(shapefile: str = None):
    """
    运行Gender Norm Index可视化 (报纸数据版)

    Args:
        shapefile: 中国地图shapefile路径(可选)
    """
    print(f"\n{'#'*70}")
    print(f"# Gender Norm Index 可视化器 (报纸数据版)")
    print(f"# 数据来源: {DATA_SOURCE}")
    print(f"# 数据年份: {DATA_YEAR}")
    print(f"{'#'*70}\n")

    print(f"加载数据...")
    data = load_results()

    if data["projections"] is None and data["weat"] is None:
        print(f"[错误] 未找到分析结果, 请先运行 newspaper_gender_norm_index_builder.py")
        return

    plot_projection_boxplot(data["projections"])

    plot_weat_distribution(data["weat"])

    plot_weat_ranking(data["weat"])

    plot_weat_heatmap(data["weat"])

    plot_category_comparison(data["projections"])

    plot_province_map(data["weat"], shapefile)

    print(f"\n{'#'*70}")
    print(f"# 可视化完成!")
    print(f"# 输出目录: {os.path.join(INPUT_DIR, 'figures')}/")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
