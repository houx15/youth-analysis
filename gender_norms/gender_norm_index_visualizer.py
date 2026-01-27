"""
Gender Norm Index 可视化器

按照 plan.md 的规范生成可视化：
1. 跨省可比性诊断图：各省所有概念词投影值的boxplot
2. 三个index的分布图：各省效应量的histogram或density plot
3. 地图可视化：三个index的省级choropleth map

输入：gender_embedding/results/gender_norm_index/{year}/ 下的分析结果
输出：gender_embedding/results/gender_norm_index/{year}/figures/ 下的PDF图表
"""

import os
import glob
import warnings
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# 尝试导入geopandas（可选）
try:
    import geopandas as gpd

    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    print("[警告] geopandas未安装，将跳过地图可视化")

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


# =============================================================================
# 配置
# =============================================================================

INPUT_DIR = "gender_norms/gender_embedding/results/gender_norm_index"
SHAPEFILE_DIR = "configs/china_shp"

# 省份名称映射（用于地图）
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

# 省份编码映射（2020年数据使用）
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
}


# =============================================================================
# 数据加载
# =============================================================================


def load_results(year: int):
    """加载分析结果"""
    year_input_dir = os.path.join(INPUT_DIR, str(year))

    files = {
        "projections": os.path.join(year_input_dir, "word_projections.csv"),
        "weat": os.path.join(year_input_dir, "weat_results.csv"),
        "index": os.path.join(year_input_dir, "gender_norm_index.csv"),
        "stats": os.path.join(year_input_dir, "province_projection_stats.csv"),
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


def generate_output_path(year: int, filename: str) -> str:
    """生成输出文件路径"""
    year_dir = os.path.join(INPUT_DIR, str(year), "figures")
    os.makedirs(year_dir, exist_ok=True)

    date_prefix = datetime.now().strftime("%Y%m%d")
    return os.path.join(year_dir, f"{date_prefix}_{filename}.pdf")


# =============================================================================
# 可视化函数
# =============================================================================


def plot_projection_boxplot(projection_df: pd.DataFrame, year: int):
    """
    1. 跨省可比性诊断图：各省所有概念词投影值的boxplot

    用于判断是否需要省内标准化
    """
    print(f"\n  生成跨省可比性诊断图...")

    if projection_df is None or len(projection_df) == 0:
        print(f"    [跳过] 无投影数据")
        return

    # 按省份排序
    province_order = (
        projection_df.groupby("province")["cosine_sim"]
        .median()
        .sort_values()
        .index.tolist()
    )

    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    # 1. 所有概念词的投影分布（boxplot）
    ax1 = axes[0]
    sns.boxplot(
        data=projection_df,
        x="province",
        y="cosine_sim",
        order=province_order,
        ax=ax1,
        palette="Set3",
    )
    ax1.axhline(y=0, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax1.set_xlabel("省份", fontsize=12, fontweight="bold")
    ax1.set_ylabel("余弦相似度（性别轴投影）", fontsize=12, fontweight="bold")
    ax1.set_title(
        f"各省概念词在性别轴上的投影分布 ({year}年)\n"
        "正值=偏女性方向，负值=偏男性方向",
        fontsize=14,
        fontweight="bold",
    )
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(axis="y", alpha=0.3)

    # 2. 分类别的投影分布（violin plot）
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
        "family": "家庭",
        "work": "工作",
        "leadership": "领导力",
        "non_leadership": "非领导力",
        "stem": "STEM",
        "non_stem": "非STEM",
    }

    # 只保留存在的类别
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
    ax2.set_xlabel("词类别", fontsize=12, fontweight="bold")
    ax2.set_ylabel("余弦相似度（性别轴投影）", fontsize=12, fontweight="bold")
    ax2.set_title(
        f"各类别概念词的投影分布 ({year}年)",
        fontsize=14,
        fontweight="bold",
    )
    ax2.set_xticklabels([category_labels.get(c, c) for c in existing_categories])
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_file = generate_output_path(year, "comparability_diagnostic")
    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    print(f"    [保存] {output_file}")
    plt.close()


def plot_weat_distribution(weat_df: pd.DataFrame, year: int):
    """
    2. 三个index的分布图：各省效应量的histogram或density plot
    """
    print(f"\n  生成WEAT效应量分布图...")

    if weat_df is None or len(weat_df) == 0:
        print(f"    [跳过] 无WEAT数据")
        return

    dimensions = ["work_family", "leadership", "stem"]
    dim_labels = {
        "work_family": "工作-家庭维度\n(正值=家庭偏女性)",
        "leadership": "领导力维度\n(正值=领导偏男性)",
        "stem": "STEM维度\n(正值=STEM偏男性)",
    }
    colors = ["#2ecc71", "#e74c3c", "#3498db"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, dim in enumerate(dimensions):
        ax = axes[i]
        dim_data = weat_df[weat_df["dimension"] == dim]["cohens_d"]

        if len(dim_data) == 0:
            ax.text(
                0.5, 0.5, "无数据", ha="center", va="center", transform=ax.transAxes
            )
            continue

        # 绘制histogram + KDE
        sns.histplot(
            dim_data,
            kde=True,
            ax=ax,
            color=colors[i],
            alpha=0.6,
            edgecolor="white",
        )

        # 添加均值和中位数线
        mean_val = dim_data.mean()
        median_val = dim_data.median()
        ax.axvline(
            x=mean_val,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"均值: {mean_val:.3f}",
        )
        ax.axvline(
            x=median_val,
            color="blue",
            linestyle=":",
            linewidth=2,
            label=f"中位数: {median_val:.3f}",
        )
        ax.axvline(x=0, color="black", linestyle="-", linewidth=1, alpha=0.5)

        ax.set_xlabel("Cohen's d", fontsize=12, fontweight="bold")
        ax.set_ylabel("频数", fontsize=12, fontweight="bold")
        ax.set_title(dim_labels[dim], fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle(
        f"Gender Norm Index 分布 ({year}年)\n|d|≈0.2小效应, |d|≈0.5中效应, |d|≈0.8大效应",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    output_file = generate_output_path(year, "weat_distribution")
    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    print(f"    [保存] {output_file}")
    plt.close()


def plot_weat_ranking(weat_df: pd.DataFrame, year: int):
    """
    各省份WEAT效应量排名条形图
    """
    print(f"\n  生成WEAT效应量排名图...")

    if weat_df is None or len(weat_df) == 0:
        print(f"    [跳过] 无WEAT数据")
        return

    dimensions = ["work_family", "leadership", "stem"]
    dim_labels = {
        "work_family": "工作-家庭维度 (d)",
        "leadership": "领导力维度 (d)",
        "stem": "STEM维度 (d)",
    }
    colors = ["#2ecc71", "#e74c3c", "#3498db"]

    for dim, color in zip(dimensions, colors):
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        dim_data = weat_df[weat_df["dimension"] == dim].copy()
        if len(dim_data) == 0:
            plt.close()
            continue

        dim_data = dim_data.sort_values("cohens_d", ascending=True)

        # 根据正负值设置颜色
        bar_colors = [color if d >= 0 else "#95a5a6" for d in dim_data["cohens_d"]]

        bars = ax.barh(
            dim_data["province"],
            dim_data["cohens_d"],
            color=bar_colors,
            alpha=0.7,
            edgecolor="white",
        )

        # 添加数值标签
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
            f"各省份 {dim_labels[dim]} 排名 ({year}年)",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        output_file = generate_output_path(year, f"weat_ranking_{dim}")
        plt.savefig(output_file, format="pdf", bbox_inches="tight")
        print(f"    [保存] {output_file}")
        plt.close()


def plot_weat_heatmap(weat_df: pd.DataFrame, year: int):
    """
    WEAT效应量热力图：省份 × 维度
    """
    print(f"\n  生成WEAT效应量热力图...")

    if weat_df is None or len(weat_df) == 0:
        print(f"    [跳过] 无WEAT数据")
        return

    # 转换为宽格式
    pivot_df = weat_df.pivot_table(
        index="province",
        columns="dimension",
        values="cohens_d",
        aggfunc="first",
    )

    # 按work_family排序
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

    ax.set_xlabel("维度", fontsize=12, fontweight="bold")
    ax.set_ylabel("省份", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Gender Norm Index 热力图 ({year}年)\n正值=传统性别规范方向",
        fontsize=14,
        fontweight="bold",
    )

    # 修改列标签
    col_labels = {
        "work_family": "工作-家庭",
        "leadership": "领导力",
        "stem": "STEM",
    }
    ax.set_xticklabels([col_labels.get(c, c) for c in pivot_df.columns])

    plt.tight_layout()
    output_file = generate_output_path(year, "weat_heatmap")
    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    print(f"    [保存] {output_file}")
    plt.close()


def plot_province_map(
    weat_df: pd.DataFrame,
    year: int,
    shapefile_path: Optional[str] = None,
):
    """
    3. 地图可视化：三个index的省级choropleth map
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
        print(f"    [跳过] 无法识别省份名称列")
        return

    dimensions = ["work_family", "leadership", "stem"]
    dim_labels = {
        "work_family": "工作-家庭维度",
        "leadership": "领导力维度",
        "stem": "STEM维度",
    }

    for dim in dimensions:
        dim_data = weat_df[weat_df["dimension"] == dim][["province", "cohens_d"]].copy()

        if len(dim_data) == 0:
            continue

        # 映射省份名称
        dim_data["province_full"] = dim_data["province"].map(PROVINCE_NAME_MAPPING)
        dim_data["province_full"] = dim_data["province_full"].fillna(
            dim_data["province"]
        )

        # 合并数据
        china_map_merged = china_map.merge(
            dim_data, left_on=name_col, right_on="province_full", how="left"
        )

        # 绘制地图
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))

        # 使用红绿配色
        china_map_merged.plot(
            column="cohens_d",
            cmap="RdYlGn",
            linewidth=0.5,
            edgecolor="white",
            legend=True,
            ax=ax,
            missing_kwds={"color": "lightgrey", "label": "无数据"},
            legend_kwds={
                "label": f"Cohen's d ({dim_labels[dim]})",
                "orientation": "vertical",
                "shrink": 0.6,
            },
        )

        # 添加省份标签
        for idx, row in china_map_merged.iterrows():
            if pd.notna(row.get("cohens_d")):
                centroid = row["geometry"].centroid
                ax.annotate(
                    text=f"{row['province']}\n{row['cohens_d']:+.2f}",
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
            f"Gender Norm Index: {dim_labels[dim]} ({year}年)\n"
            "正值=传统性别规范方向（颜色越绿=越传统）",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax.axis("off")

        plt.tight_layout()
        output_file = generate_output_path(year, f"map_{dim}")
        plt.savefig(output_file, format="pdf", bbox_inches="tight")
        print(f"    [保存] {output_file}")
        plt.close()


def plot_category_comparison(projection_df: pd.DataFrame, year: int):
    """
    分类别的省份间比较图
    """
    print(f"\n  生成类别间比较图...")

    if projection_df is None or len(projection_df) == 0:
        print(f"    [跳过] 无投影数据")
        return

    # 计算每个省份每个类别的平均投影
    category_means = (
        projection_df.groupby(["province", "category"])["cosine_sim"]
        .mean()
        .reset_index()
    )

    # 对比维度
    comparisons = [
        ("family", "work", "家庭 vs 工作"),
        ("non_leadership", "leadership", "非领导 vs 领导"),
        ("non_stem", "stem", "非STEM vs STEM"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    for i, (cat1, cat2, title) in enumerate(comparisons):
        ax = axes[i]

        cat1_data = category_means[category_means["category"] == cat1][
            ["province", "cosine_sim"]
        ]
        cat2_data = category_means[category_means["category"] == cat2][
            ["province", "cosine_sim"]
        ]

        if len(cat1_data) == 0 or len(cat2_data) == 0:
            ax.text(
                0.5, 0.5, "无数据", ha="center", va="center", transform=ax.transAxes
            )
            continue

        merged = cat1_data.merge(cat2_data, on="province", suffixes=("_1", "_2"))

        ax.scatter(
            merged["cosine_sim_2"],
            merged["cosine_sim_1"],
            alpha=0.7,
            s=100,
            c="#3498db",
            edgecolors="white",
        )

        # 添加省份标签
        for _, row in merged.iterrows():
            ax.annotate(
                row["province"],
                (row["cosine_sim_2"], row["cosine_sim_1"]),
                fontsize=8,
                alpha=0.8,
            )

        # 添加对角线
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1)
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)

        ax.set_xlabel(f"{cat2} 平均投影", fontsize=11, fontweight="bold")
        ax.set_ylabel(f"{cat1} 平均投影", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(alpha=0.3)

    plt.suptitle(
        f"类别间性别偏向比较 ({year}年)\n点在对角线上方=第一类别更偏女性",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    output_file = generate_output_path(year, "category_comparison")
    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    print(f"    [保存] {output_file}")
    plt.close()


# =============================================================================
# 主函数
# =============================================================================


def main(year: int, shapefile: str = None):
    """
    运行Gender Norm Index可视化

    Args:
        year: 年份
        shapefile: 中国地图shapefile路径（可选）
    """
    print(f"\n{'#'*70}")
    print(f"# Gender Norm Index 可视化器")
    print(f"# 年份: {year}")
    print(f"{'#'*70}\n")

    # 加载数据
    print(f"加载数据...")
    data = load_results(year)

    if data["projections"] is None and data["weat"] is None:
        print(f"[错误] 未找到分析结果，请先运行 gender_norm_index_builder.py")
        return

    # 1. 跨省可比性诊断图
    plot_projection_boxplot(data["projections"], year)

    # 2. WEAT效应量分布图
    plot_weat_distribution(data["weat"], year)

    # 3. WEAT效应量排名图
    plot_weat_ranking(data["weat"], year)

    # 4. WEAT热力图
    plot_weat_heatmap(data["weat"], year)

    # 5. 类别间比较图
    plot_category_comparison(data["projections"], year)

    # 6. 地图可视化
    plot_province_map(data["weat"], year, shapefile)

    print(f"\n{'#'*70}")
    print(f"# 可视化完成!")
    print(f"# 输出目录: {os.path.join(INPUT_DIR, str(year), 'figures')}/")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
