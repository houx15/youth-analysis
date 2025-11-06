"""
省份性别-职业偏向可视化分析（使用geopandas绘制地图）

功能：
1. 省份性别隔离程度地图（使用geopandas）
2. 省份聚类分析（基于职业性别偏向模式）
3. 特定职业的省份差异对比
4. 省份间模式相似度分析

输入数据：embedding_analysis/ 目录下的分析结果
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
import warnings

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

INPUT_DIR = "embedding_analysis"
OUTPUT_DIR = "embedding_visualization"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 省份编码映射（GB/T 2260 中华人民共和国行政区划代码）
# 如果analyzer输出的省份是编码格式，将编码转换为省份名称
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
    # 处理可能的非标准编码
    "100": "未知",
    "400": "未知",
}

# 省份名称标准化映射（处理shapefile中的命名差异）
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

# 省份到地理区域的映射
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


def load_results(year):
    """加载分析结果"""
    stats_file = os.path.join(INPUT_DIR, f"province_stats_{year}.csv")
    occupation_file = os.path.join(INPUT_DIR, f"occupation_bias_{year}.csv")

    if not os.path.exists(stats_file) or not os.path.exists(occupation_file):
        print(f"❌ 未找到 {year} 年的分析结果")
        return None, None

    stats_df = pd.read_csv(stats_file)
    occupation_df = pd.read_csv(occupation_file)

    # 将省份编码转换为省份名称（如果analyzer输出的是编码格式）
    def convert_province_code(province):
        """将省份编码转换为省份名称"""
        if pd.isna(province):
            return province
        # 统一转换为字符串格式处理
        if isinstance(province, (int, float)):
            code_str = str(int(province))  # 去掉小数点
        else:
            code_str = str(province).strip()

        # 如果是编码，转换为名称
        if code_str in PROVINCE_CODE_TO_NAME:
            return PROVINCE_CODE_TO_NAME[code_str]
        # 如果已经是名称，直接返回
        elif code_str in PROVINCE_TO_REGION:
            return code_str
        # 如果都不匹配，返回原值
        return code_str

    # 转换省份编码
    print(f"  正在检查并转换省份编码...")
    original_provinces = set(stats_df["province"].unique())

    # 统计有多少是编码格式
    code_count = sum(
        1 for p in original_provinces if str(p).strip() in PROVINCE_CODE_TO_NAME
    )
    name_count = len(original_provinces) - code_count

    if code_count > 0:
        print(f"  发现 {code_count} 个编码格式的省份，{name_count} 个名称格式的省份")

    stats_df["province"] = stats_df["province"].apply(convert_province_code)
    occupation_df["province"] = occupation_df["province"].apply(convert_province_code)

    # 检查转换结果
    unique_provinces = stats_df["province"].unique()
    print(
        f"  转换后的省份: {', '.join(sorted(unique_provinces)[:15])}{'...' if len(unique_provinces) > 15 else ''}"
    )

    # 检查是否有未识别的省份
    unknown_provinces = [p for p in unique_provinces if p not in PROVINCE_TO_REGION]
    if unknown_provinces:
        print(f"  ⚠️  以下省份未在区域映射中找到: {', '.join(unknown_provinces)}")
        print(f"     这些省份可能来自非标准编码，将标记为'未知区域'")

    # 添加地理区域信息
    stats_df["region"] = stats_df["province"].map(PROVINCE_TO_REGION)
    occupation_df["region"] = occupation_df["province"].map(PROVINCE_TO_REGION)

    # 处理未识别的省份
    stats_df["region"] = stats_df["region"].fillna("未知区域")
    occupation_df["region"] = occupation_df["region"].fillna("未知区域")

    print(f"✓ 加载了 {len(stats_df)} 个省份的数据")
    print(f"✓ 加载了 {len(occupation_df)} 条职业-省份记录")

    return stats_df, occupation_df


def load_china_map(shapefile_path=None):
    """
    加载中国地图shapefile

    Args:
        shapefile_path: shapefile路径或文件夹路径（如果为None，自动查找configs/china_shp文件夹）

    Returns:
        GeoDataFrame 或 None（如果加载失败）
    """
    # 如果没有指定路径，尝试从configs/china_shp文件夹加载
    if shapefile_path is None:
        shapefile_dir = "configs/china_shp"
        if os.path.exists(shapefile_dir) and os.path.isdir(shapefile_dir):
            # 查找文件夹中的.shp文件
            shp_files = glob.glob(os.path.join(shapefile_dir, "*.shp"))
            if shp_files:
                shapefile_path = shp_files[0]  # 使用找到的第一个.shp文件
                print(f"自动找到地图文件: {shapefile_path}")
            else:
                print(f"⚠️  在 {shapefile_dir} 中未找到.shp文件")
                return None
        else:
            print(f"⚠️  地图文件夹不存在: {shapefile_dir}")
            return None

    # 如果是文件夹路径，查找其中的.shp文件
    if os.path.isdir(shapefile_path):
        shp_files = glob.glob(os.path.join(shapefile_path, "*.shp"))
        if shp_files:
            shapefile_path = shp_files[0]
        else:
            print(f"⚠️  在 {shapefile_path} 中未找到.shp文件")
            return None

    if not os.path.exists(shapefile_path):
        print(f"⚠️  地图文件不存在: {shapefile_path}")
        return None

    try:
        print(f"正在加载地图文件: {shapefile_path}")
        gdf = gpd.read_file(shapefile_path)

        if gdf.empty:
            print(f"⚠️  地图文件为空")
            return None

        print(f"✓ 成功加载，包含 {len(gdf)} 个地理要素")
        print(f"  地图列名: {gdf.columns.tolist()}")

        # 确保使用正确的CRS
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326", allow_override=True)

        return gdf

    except Exception as e:
        print(f"❌ 加载地图文件失败: {e}")
        return None


def plot_china_map_segregation(stats_df, year, shapefile_path=None):
    """
    使用geopandas绘制中国地图：展示各省份的性别隔离程度

    Args:
        stats_df: 省份统计数据
        year: 年份
        shapefile_path: 中国地图shapefile路径（可选）
    """
    # 加载地图（自动从configs/china_shp文件夹加载）
    china_map = load_china_map(shapefile_path)

    if china_map is None:
        print("❌ 无法加载地图文件，跳过地图绘制")
        print("   将绘制替代图表...")
        plot_static_alternatives(stats_df, year)
        return

    # 打印shapefile的列名，帮助调试
    print(f"  Shapefile列名: {china_map.columns.tolist()}")

    # 自动识别省份名称列（humdata adm1数据通常使用ADMIN1或NAME_1）
    possible_name_cols = [
        "ADMIN1",  # humdata标准列名
        "admin1",
        "NAME_1",  # humdata常用列名
        "name_1",
        "NAME",  # 其他可能的列名
        "name",
        "PROV",
        "prov",
        "Province",
        "province",
        "NAME_CH",
        "name_ch",
        "FCNAME",  # 中文名称
        "fcname",
    ]
    name_col = None
    for col in possible_name_cols:
        if col in china_map.columns:
            name_col = col
            break

    if name_col is None:
        print(f"⚠️  无法自动识别省份名称列，请手动指定")
        print(f"   可用列: {china_map.columns.tolist()}")
        return

    print(f"  使用省份名称列: {name_col}")

    # 标准化省份名称
    stats_df_copy = stats_df.copy()
    stats_df_copy["province_full"] = stats_df_copy["province"].map(
        PROVINCE_NAME_MAPPING
    )

    # 如果mapping后还是None，说明就是原名
    stats_df_copy["province_full"] = stats_df_copy["province_full"].fillna(
        stats_df_copy["province"]
    )

    # 合并数据
    china_map_merged = china_map.merge(
        stats_df_copy, left_on=name_col, right_on="province_full", how="left"
    )

    # 检查合并情况
    matched = china_map_merged["std_bias"].notna().sum()
    total_provinces = len(stats_df)
    print(f"  地图匹配: {matched}/{total_provinces} 个省份")

    if matched == 0:
        print("⚠️  没有匹配到任何省份，可能是命名不一致")
        print(f"  地图中的省份名称示例: {china_map[name_col].head().tolist()}")
        print(
            f"  数据中的省份名称示例: {stats_df_copy['province_full'].head().tolist()}"
        )
        return

    # 绘制地图
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    # 自定义配色方案（白色->橙色->红色->深红色）
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

    # 绘制有数据的省份
    china_map_merged.plot(
        column="std_bias",
        cmap=cmap,
        linewidth=0.5,
        edgecolor="white",
        legend=True,
        ax=ax,
        missing_kwds={"color": "lightgrey", "label": "无数据"},
        legend_kwds={
            "label": "性别隔离指数（标准差）",
            "orientation": "vertical",
            "shrink": 0.6,
            "pad": 0.05,
        },
    )

    # 添加省份标签（只标注有数据的省份）
    for idx, row in china_map_merged.iterrows():
        if pd.notna(row["std_bias"]):
            # 获取省份中心点
            centroid = row["geometry"].centroid

            # 标注省份名称和数值
            ax.annotate(
                text=f"{row['province']}\n{row['std_bias']:.3f}",
                xy=(centroid.x, centroid.y),
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="black",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    alpha=0.7,
                    edgecolor="none",
                ),
            )

    ax.set_title(
        f"中国各省份职业性别隔离程度地图 ({year}年)\n"
        + "颜色越深 = 性别隔离程度越高（职业性别分化越明显）",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.axis("off")

    # 添加统计信息文本框
    stats_text = (
        f"分析省份数: {total_provinces}\n"
        f"最高: {stats_df.nlargest(1, 'std_bias')['province'].values[0]} ({stats_df['std_bias'].max():.3f})\n"
        f"最低: {stats_df.nsmallest(1, 'std_bias')['province'].values[0]} ({stats_df['std_bias'].min():.3f})\n"
        f"平均: {stats_df['std_bias'].mean():.3f}"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    map_file = os.path.join(OUTPUT_DIR, f"segregation_map_{year}.pdf")
    plt.savefig(map_file, format="pdf", bbox_inches="tight")
    print(f"✓ 中国地图已保存: {map_file}")
    plt.close()

    # 绘制第二张地图：按区域着色
    plot_regional_map(china_map, china_map_merged, stats_df, year, name_col)


def plot_regional_map(china_map, china_map_merged, stats_df, year, name_col):
    """绘制按地理区域着色的地图"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    # 为每个区域分配颜色
    region_colors = {
        "华北": "#e41a1c",
        "东北": "#377eb8",
        "华东": "#4daf4a",
        "华中": "#984ea3",
        "华南": "#ff7f00",
        "西南": "#ffff33",
        "西北": "#a65628",
    }

    # 添加区域颜色到地图数据
    china_map_merged["region_color"] = china_map_merged["region"].map(region_colors)

    # 绘制地图
    china_map_merged.plot(
        color=china_map_merged["region_color"].fillna("lightgrey"),
        linewidth=0.5,
        edgecolor="white",
        ax=ax,
        alpha=0.6,
    )

    # 添加省份标签和数值
    for idx, row in china_map_merged.iterrows():
        if pd.notna(row["std_bias"]):
            centroid = row["geometry"].centroid
            ax.annotate(
                text=f"{row['province']}\n{row['std_bias']:.3f}",
                xy=(centroid.x, centroid.y),
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="black",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    alpha=0.9,
                    edgecolor="none",
                ),
            )

    ax.set_title(
        f"中国各省份性别隔离程度：按地理区域分类 ({year}年)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.axis("off")

    # 添加图例
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(
            facecolor=color,
            label=f'{region} (均值: {stats_df[stats_df["region"]==region]["std_bias"].mean():.3f})',
        )
        for region, color in region_colors.items()
        if region in stats_df["region"].values
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower left",
        fontsize=10,
        title="地理区域",
        title_fontsize=11,
        framealpha=0.9,
    )

    plt.tight_layout()
    regional_map_file = os.path.join(OUTPUT_DIR, f"segregation_map_regional_{year}.pdf")
    plt.savefig(regional_map_file, format="pdf", bbox_inches="tight")
    print(f"✓ 区域地图已保存: {regional_map_file}")
    plt.close()


def plot_static_alternatives(stats_df, year):
    """如果无法加载地图，绘制替代图表"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # 按区域分组
    region_data = []
    for region, provinces in PROVINCE_REGIONS.items():
        region_provinces = stats_df[stats_df["province"].isin(provinces)]
        if len(region_provinces) > 0:
            region_data.append(
                {
                    "region": region,
                    "mean_segregation": region_provinces["std_bias"].mean(),
                    "provinces": ", ".join(region_provinces["province"].tolist()),
                }
            )

    region_df = pd.DataFrame(region_data).sort_values(
        "mean_segregation", ascending=False
    )

    # 绘制柱状图
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(region_df)))
    bars = ax.barh(region_df["region"], region_df["mean_segregation"], color=colors)

    for i, (bar, row) in enumerate(zip(bars, region_df.itertuples())):
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{row.mean_segregation:.3f}",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel("平均性别隔离指数（标准差）", fontsize=12, fontweight="bold")
    ax.set_title(
        f"中国各地区职业性别隔离程度 ({year}年)\n数值越大 = 职业性别分化越明显",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    plt.tight_layout()
    map_file = os.path.join(OUTPUT_DIR, f"segregation_by_region_{year}.pdf")
    plt.savefig(map_file, format="pdf", bbox_inches="tight")
    print(f"✓ 区域柱状图已保存: {map_file}")
    plt.close()


def plot_province_ranking(stats_df, year):
    """绘制详细的省份排名图"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    stats_sorted = stats_df.sort_values("std_bias", ascending=True)

    # 按区域着色
    colors = [
        plt.cm.Set3(
            list(PROVINCE_REGIONS.keys()).index(PROVINCE_TO_REGION.get(p, "华北")) / 7
        )
        for p in stats_sorted["province"]
    ]

    bars = ax.barh(stats_sorted["province"], stats_sorted["std_bias"], color=colors)

    # 添加数值标签
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}",
            va="center",
            fontsize=9,
        )

    ax.set_xlabel("性别隔离指数（标准差）", fontsize=12, fontweight="bold")
    ax.set_title(
        f"各省份职业性别隔离程度排名 ({year}年)", fontsize=14, fontweight="bold", pad=20
    )
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    # 添加图例
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=plt.cm.Set3(i / 7), label=region)
        for i, region in enumerate(PROVINCE_REGIONS.keys())
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9, title="地理区域")

    plt.tight_layout()
    ranking_file = os.path.join(OUTPUT_DIR, f"segregation_ranking_{year}.pdf")
    plt.savefig(ranking_file, format="pdf", bbox_inches="tight")
    print(f"✓ 省份排名图已保存: {ranking_file}")
    plt.close()


def plot_province_clustering(occupation_df, stats_df, year):
    """省份聚类分析：基于职业性别偏向模式"""
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist, squareform

    # 创建省份×职业矩阵
    pivot = occupation_df.pivot_table(
        values="bias_score", index="province", columns="occupation", aggfunc="mean"
    ).fillna(0)

    # 层次聚类
    linkage_matrix = linkage(pivot, method="ward")

    # 绘制树状图
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    dendrogram(
        linkage_matrix,
        labels=pivot.index.tolist(),
        leaf_font_size=11,
        ax=ax,
        color_threshold=0.7 * max(linkage_matrix[:, 2]),
    )

    ax.set_title(
        f"省份性别观念模式聚类分析 ({year}年)\n基于职业性别偏向模式的相似度",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("省份", fontsize=12, fontweight="bold")
    ax.set_ylabel("距离（差异程度）", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    cluster_file = os.path.join(OUTPUT_DIR, f"province_clustering_{year}.pdf")
    plt.savefig(cluster_file, format="pdf", bbox_inches="tight")
    print(f"✓ 省份聚类图已保存: {cluster_file}")
    plt.close()

    # 绘制热力图：省份相似度矩阵
    distances = pdist(pivot, metric="euclidean")
    distance_matrix = squareform(distances)

    # 转换为相似度
    max_dist = distance_matrix.max()
    similarity_matrix = 1 - (distance_matrix / max_dist)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    sns.heatmap(
        similarity_matrix,
        xticklabels=pivot.index,
        yticklabels=pivot.index,
        annot=False,
        fmt=".2f",
        cmap="YlOrRd",
        cbar_kws={"label": "模式相似度"},
        ax=ax,
        square=True,
    )

    ax.set_title(
        f"省份性别观念模式相似度矩阵 ({year}年)\n颜色越深 = 模式越相似",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()
    similarity_file = os.path.join(OUTPUT_DIR, f"province_similarity_{year}.pdf")
    plt.savefig(similarity_file, format="pdf", bbox_inches="tight")
    print(f"✓ 省份相似度矩阵已保存: {similarity_file}")
    plt.close()


def plot_province_comparison(stats_df, year):
    """省份多维度对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 隔离程度 vs 平均偏向
    ax = axes[0, 0]
    scatter = ax.scatter(
        stats_df["mean_bias"],
        stats_df["std_bias"],
        s=stats_df["text_count"] / 1000,
        c=stats_df["std_bias"],
        cmap="Reds",
        alpha=0.6,
        edgecolors="black",
        linewidth=1,
    )

    for _, row in stats_df.iterrows():
        ax.annotate(
            row["province"],
            (row["mean_bias"], row["std_bias"]),
            fontsize=8,
            ha="center",
        )

    ax.axhline(
        y=stats_df["std_bias"].mean(),
        color="gray",
        linestyle="--",
        alpha=0.5,
        label="平均隔离程度",
    )
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5, label="性别中性")

    ax.set_xlabel(
        "平均性别偏向\n(负=偏男性, 正=偏女性)", fontsize=11, fontweight="bold"
    )
    ax.set_ylabel(
        "性别隔离指数\n(标准差，值越大=隔离越明显)", fontsize=11, fontweight="bold"
    )
    ax.set_title("省份性别观念二维分布", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    # 2. 隔离程度排名（Top 15）
    ax = axes[0, 1]
    top_15 = stats_df.nlargest(15, "std_bias").sort_values("std_bias")
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(top_15)))
    bars = ax.barh(top_15["province"], top_15["std_bias"], color=colors)

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}",
            va="center",
            fontsize=9,
        )

    ax.set_xlabel("性别隔离指数", fontsize=11, fontweight="bold")
    ax.set_title("性别隔离最明显的省份 (Top 15)", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # 3. 数据质量分布
    ax = axes[1, 0]
    stats_sorted = stats_df.sort_values("text_count", ascending=False)
    bars = ax.bar(
        range(len(stats_sorted)),
        stats_sorted["text_count"] / 10000,
        color="steelblue",
        alpha=0.7,
    )
    ax.set_xticks(range(len(stats_sorted)))
    ax.set_xticklabels(stats_sorted["province"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("文本数量（万条）", fontsize=11, fontweight="bold")
    ax.set_title("各省份数据量分布", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # 4. 区域对比箱线图
    ax = axes[1, 1]
    region_order = ["华北", "东北", "华东", "华中", "华南", "西南", "西北"]
    data_by_region = [
        stats_df[stats_df["region"] == r]["std_bias"].values
        for r in region_order
        if r in stats_df["region"].values
    ]
    labels_with_data = [r for r in region_order if r in stats_df["region"].values]

    bp = ax.boxplot(data_by_region, labels=labels_with_data, patch_artist=True)
    for patch, color in zip(
        bp["boxes"], plt.cm.Set3(np.linspace(0, 1, len(data_by_region)))
    ):
        patch.set_facecolor(color)

    ax.set_ylabel("性别隔离指数", fontsize=11, fontweight="bold")
    ax.set_title("各地理区域性别隔离程度分布", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    comparison_file = os.path.join(OUTPUT_DIR, f"province_comparison_{year}.pdf")
    plt.savefig(comparison_file, format="pdf", bbox_inches="tight")
    print(f"✓ 省份对比图已保存: {comparison_file}")
    plt.close()


def plot_occupation_by_province(occupation_df, occupation_name, year):
    """特定职业在各省份的性别偏向对比"""
    occ_data = occupation_df[occupation_df["occupation"] == occupation_name].copy()

    if len(occ_data) == 0:
        print(f"⚠️  未找到职业: {occupation_name}")
        return

    occ_data = occ_data.sort_values("bias_score")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 左图：性别偏向分数
    colors = ["#d62728" if x < 0 else "#2ca02c" for x in occ_data["bias_score"]]
    bars = ax1.barh(
        occ_data["province"], occ_data["bias_score"], color=colors, alpha=0.7
    )

    ax1.axvline(x=0, color="black", linestyle="--", linewidth=1)
    ax1.set_xlabel(
        "性别偏向分数\n(负=偏男性, 正=偏女性)", fontsize=11, fontweight="bold"
    )
    ax1.set_title(
        f'"{occupation_name}"的性别关联：各省份差异', fontsize=12, fontweight="bold"
    )
    ax1.grid(axis="x", alpha=0.3)

    for bar in bars:
        width = bar.get_width()
        ax1.text(
            width + (0.005 if width > 0 else -0.005),
            bar.get_y() + bar.get_height() / 2,
            f"{width:+.3f}",
            va="center",
            ha="left" if width > 0 else "right",
            fontsize=8,
        )

    # 右图：男性/女性相似度对比
    x = np.arange(len(occ_data))
    width = 0.35

    bars1 = ax2.barh(
        x - width / 2,
        occ_data["male_similarity"],
        width,
        label="男性相似度",
        color="#1f77b4",
        alpha=0.7,
    )
    bars2 = ax2.barh(
        x + width / 2,
        occ_data["female_similarity"],
        width,
        label="女性相似度",
        color="#ff7f0e",
        alpha=0.7,
    )

    ax2.set_yticks(x)
    ax2.set_yticklabels(occ_data["province"])
    ax2.set_xlabel("与性别词的相似度", fontsize=11, fontweight="bold")
    ax2.set_title(
        f'"{occupation_name}"与性别词的相似度分解', fontsize=12, fontweight="bold"
    )
    ax2.legend(fontsize=10)
    ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    occ_file = os.path.join(OUTPUT_DIR, f"occupation_{occupation_name}_{year}.pdf")
    plt.savefig(occ_file, format="pdf", bbox_inches="tight")
    print(f"✓ 职业分析图已保存: {occ_file}")
    plt.close()


def generate_summary_report(stats_df, occupation_df, year):
    """生成可视化分析总结报告"""
    report_file = os.path.join(OUTPUT_DIR, f"visualization_summary_{year}.txt")

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"{'='*70}\n")
        f.write(f"省份性别-职业偏向可视化分析总结 ({year}年)\n")
        f.write(f"{'='*70}\n\n")

        # 1. 性别隔离程度排名
        f.write(f"{'='*70}\n")
        f.write(f"一、性别隔离程度排名（标准差）\n")
        f.write(f"{'='*70}\n")
        f.write(f"说明：标准差越大 = 职业性别分化越明显 = 性别隔离越严重\n\n")

        stats_sorted = stats_df.sort_values("std_bias", ascending=False)
        f.write(f"Top 10 性别隔离最明显的省份:\n")
        for i, (_, row) in enumerate(stats_sorted.head(10).iterrows(), 1):
            f.write(
                f"  {i:2d}. {row['province']:8s} | "
                f"隔离指数: {row['std_bias']:.4f} | "
                f"平均偏向: {row['mean_bias']:+.4f} | "
                f"区域: {row['region']}\n"
            )

        f.write(f"\nTop 10 性别隔离最不明显的省份:\n")
        for i, (_, row) in enumerate(stats_sorted.tail(10).iloc[::-1].iterrows(), 1):
            f.write(
                f"  {i:2d}. {row['province']:8s} | "
                f"隔离指数: {row['std_bias']:.4f} | "
                f"平均偏向: {row['mean_bias']:+.4f} | "
                f"区域: {row['region']}\n"
            )

        # 2. 地理区域分析
        f.write(f"\n{'='*70}\n")
        f.write(f"二、地理区域分析\n")
        f.write(f"{'='*70}\n\n")

        region_stats = (
            stats_df.groupby("region")
            .agg(
                {
                    "std_bias": ["mean", "std", "min", "max"],
                    "mean_bias": "mean",
                    "province": "count",
                }
            )
            .round(4)
        )

        region_stats.columns = [
            "平均隔离",
            "隔离标准差",
            "最小隔离",
            "最大隔离",
            "平均偏向",
            "省份数",
        ]
        region_stats = region_stats.sort_values("平均隔离", ascending=False)

        f.write(region_stats.to_string())
        f.write(f"\n\n解读：\n")
        f.write(
            f"  - 平均隔离最高的区域: {region_stats.index[0]} ({region_stats.iloc[0]['平均隔离']:.4f})\n"
        )
        f.write(
            f"  - 平均隔离最低的区域: {region_stats.index[-1]} ({region_stats.iloc[-1]['平均隔离']:.4f})\n"
        )

        # 3. 极端案例分析
        f.write(f"\n{'='*70}\n")
        f.write(f"三、极端案例分析\n")
        f.write(f"{'='*70}\n\n")

        most_male_biased = stats_df.nsmallest(5, "mean_bias")
        most_female_biased = stats_df.nlargest(5, "mean_bias")

        f.write(f"整体最偏男性的省份 (Top 5):\n")
        for i, (_, row) in enumerate(most_male_biased.iterrows(), 1):
            f.write(
                f"  {i}. {row['province']:8s} | 平均偏向: {row['mean_bias']:+.4f}\n"
            )

        f.write(f"\n整体最偏女性的省份 (Top 5):\n")
        for i, (_, row) in enumerate(most_female_biased.iterrows(), 1):
            f.write(
                f"  {i}. {row['province']:8s} | 平均偏向: {row['mean_bias']:+.4f}\n"
            )

        # 4. 特定职业的省份差异
        f.write(f"\n{'='*70}\n")
        f.write(f"四、典型职业的省份差异\n")
        f.write(f"{'='*70}\n\n")

        key_occupations = ["护士", "程序员", "教师", "医生", "CEO"]
        for occ in key_occupations:
            occ_data = occupation_df[occupation_df["occupation"] == occ]
            if len(occ_data) > 0:
                f.write(f"\n【{occ}】\n")
                f.write(f"  全国平均偏向: {occ_data['bias_score'].mean():+.4f}\n")
                f.write(f"  省份间差异（标准差）: {occ_data['bias_score'].std():.4f}\n")
                f.write(
                    f"  最偏女性: {occ_data.nlargest(3, 'bias_score')['province'].tolist()}\n"
                )
                f.write(
                    f"  最偏男性: {occ_data.nsmallest(3, 'bias_score')['province'].tolist()}\n"
                )

        # 5. 数据质量说明
        f.write(f"\n{'='*70}\n")
        f.write(f"五、数据质量说明\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"  总省份数: {len(stats_df)}\n")
        f.write(f"  总文本数: {stats_df['text_count'].sum():,}\n")
        f.write(f"  平均每省份文本数: {stats_df['text_count'].mean():,.0f}\n")
        f.write(
            f"  文本数最多的省份: {stats_df.nlargest(1, 'text_count')['province'].values[0]}\n"
        )
        f.write(
            f"  文本数最少的省份: {stats_df.nsmallest(1, 'text_count')['province'].values[0]}\n"
        )

    print(f"✓ 分析总结已保存: {report_file}")


def main(year: int, shapefile: str = None):
    """
    运行可视化分析

    Args:
        year: 年份
        shapefile: 中国地图shapefile路径（可选）
                  例如: 'china_map/china_province.shp'
    """
    print(f"\n{'='*70}")
    print(f"🎨 开始生成 {year} 年省份性别-职业偏向可视化")
    print(f"{'='*70}\n")

    # 加载数据
    stats_df, occupation_df = load_results(year)
    if stats_df is None or occupation_df is None:
        return

    # 1. 中国地图（性别隔离程度）
    print(f"\n📍 生成中国地图...")
    plot_china_map_segregation(stats_df, year, shapefile)

    # 2. 省份排名图
    print(f"\n📊 生成省份排名图...")
    plot_province_ranking(stats_df, year)

    # 3. 省份聚类分析
    print(f"\n🌳 生成省份聚类分析...")
    plot_province_clustering(occupation_df, stats_df, year)

    # 4. 省份多维度对比
    print(f"\n📈 生成省份对比图...")
    plot_province_comparison(stats_df, year)

    # 5. 特定职业的省份差异
    print(f"\n👔 生成典型职业分析...")
    key_occupations = ["护士", "程序员", "教师", "医生", "CEO"]
    for occ in key_occupations:
        if occ in occupation_df["occupation"].values:
            plot_occupation_by_province(occupation_df, occ, year)

    # 6. 生成总结报告
    print(f"\n📝 生成总结报告...")
    generate_summary_report(stats_df, occupation_df, year)

    print(f"\n{'='*70}")
    print(f"✅ 可视化完成！所有文件已保存到: {OUTPUT_DIR}/")
    print(f"{'='*70}\n")

    print(f"生成的文件包括:")
    print(f"  1. segregation_map_{year}.pdf - 中国地图（性别隔离程度）")
    print(f"  2. segregation_map_regional_{year}.pdf - 中国地图（按区域着色）")
    print(f"  3. segregation_ranking_{year}.pdf - 省份排名图")
    print(f"  4. province_clustering_{year}.pdf - 省份聚类树状图")
    print(f"  5. province_similarity_{year}.pdf - 省份相似度热力图")
    print(f"  6. province_comparison_{year}.pdf - 省份多维度对比")
    print(f"  7. occupation_[职业名]_{year}.pdf - 各职业的省份分析")
    print(f"  8. visualization_summary_{year}.txt - 文字总结报告\n")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
