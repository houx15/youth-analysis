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

INPUT_DIR = "gender_embedding/results/embedding_analysis"
OUTPUT_DIR = "gender_embedding/results/embedding_visualization"
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
    # analyzer 保存的文件路径：gender_embedding/embedding_analysis/{year}/province_stats.csv
    year_input_dir = os.path.join(INPUT_DIR, str(year))
    stats_file = os.path.join(year_input_dir, "province_stats.csv")
    occupation_file = os.path.join(year_input_dir, "occupation_bias.csv")

    # 尝试加载 pivot 文件（如果存在）
    pivot_bias_file = os.path.join(year_input_dir, "occupation_bias_pivot.csv")
    pivot_projection_file = os.path.join(
        year_input_dir, "occupation_projection_pivot.csv"
    )

    # 尝试加载家务分工词汇性别轴投影数据（如果存在）
    domestic_work_projection_file = os.path.join(
        year_input_dir, "domestic_work_gender_projection.csv"
    )

    if not os.path.exists(stats_file) or not os.path.exists(occupation_file):
        print(f"❌ 未找到 {year} 年的分析结果")
        print(f"   查找路径: {stats_file}")
        print(f"   查找路径: {occupation_file}")
        return None, None, None, None, None

    stats_df = pd.read_csv(stats_file)
    occupation_df = pd.read_csv(occupation_file)

    # 加载 pivot 文件（如果存在）
    pivot_bias_df = None
    pivot_projection_df = None

    # 加载家务分工词汇性别轴投影数据（如果存在）
    domestic_work_projection_df = None

    if os.path.exists(pivot_bias_file):
        try:
            pivot_bias_df = pd.read_csv(pivot_bias_file, index_col=0)
            print(f"✓ 加载了 pivot 文件（余弦相似度差值）: {pivot_bias_file}")
        except Exception as e:
            print(f"⚠️  加载 pivot 文件失败: {e}")

    if os.path.exists(pivot_projection_file):
        try:
            pivot_projection_df = pd.read_csv(pivot_projection_file, index_col=0)
            print(f"✓ 加载了 pivot 文件（性别轴投影）: {pivot_projection_file}")
        except Exception as e:
            print(f"⚠️  加载 pivot 文件失败: {e}")

    if os.path.exists(domestic_work_projection_file):
        try:
            domestic_work_projection_df = pd.read_csv(domestic_work_projection_file)
            print(
                f"✓ 加载了家务分工词汇性别轴投影数据: {domestic_work_projection_file}"
            )
        except Exception as e:
            print(f"⚠️  加载家务分工投影数据失败: {e}")

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

    # 对家务分工投影数据的省份进行转换
    if domestic_work_projection_df is not None:
        domestic_work_projection_df["province"] = domestic_work_projection_df[
            "province"
        ].apply(convert_province_code)
        domestic_work_projection_df = domestic_work_projection_df[
            domestic_work_projection_df["province"] != "未知"
        ].copy()
        print(f"  已转换家务分工投影数据的省份名称")

    # 对 pivot 文件的列名（省份）进行转换
    if pivot_bias_df is not None:
        pivot_bias_df.columns = pivot_bias_df.columns.map(convert_province_code)
        # 过滤掉"未知"省份的列
        pivot_bias_df = pivot_bias_df.loc[:, pivot_bias_df.columns != "未知"]
        print(f"  已转换 pivot 文件（余弦相似度差值）的省份名称")

    if pivot_projection_df is not None:
        pivot_projection_df.columns = pivot_projection_df.columns.map(
            convert_province_code
        )
        # 过滤掉"未知"省份的列
        pivot_projection_df = pivot_projection_df.loc[
            :, pivot_projection_df.columns != "未知"
        ]
        print(f"  已转换 pivot 文件（性别轴投影）的省份名称")

    # 过滤掉省份名为"未知"的数据
    before_filter_stats = len(stats_df)
    before_filter_occupation = len(occupation_df)

    stats_df = stats_df[stats_df["province"] != "未知"].copy()
    occupation_df = occupation_df[occupation_df["province"] != "未知"].copy()

    filtered_stats = before_filter_stats - len(stats_df)
    filtered_occupation = before_filter_occupation - len(occupation_df)

    if filtered_stats > 0 or filtered_occupation > 0:
        print(
            f"  已过滤掉 {filtered_stats} 条统计数据和 {filtered_occupation} 条职业数据（省份为'未知'）"
        )

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
    if domestic_work_projection_df is not None:
        print(f"✓ 加载了 {len(domestic_work_projection_df)} 条家务分工投影数据")

    return (
        stats_df,
        occupation_df,
        pivot_bias_df,
        pivot_projection_df,
        domestic_work_projection_df,
    )


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
        "ADM1_ZH",  # humdata标准列名
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


def plot_province_clustering(
    occupation_df, stats_df, year, pivot_bias_df=None, pivot_projection_df=None
):
    """省份聚类分析：基于职业性别偏向模式

    Args:
        occupation_df: 长格式职业数据
        stats_df: 省份统计数据
        year: 年份
        pivot_bias_df: 预计算的 pivot 表（余弦相似度差值方法），格式为 occupation × province
        pivot_projection_df: 预计算的 pivot 表（性别轴投影方法），格式为 occupation × province
    """
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist, squareform

    # 优先使用预加载的 pivot 文件，否则从长格式数据创建
    if pivot_bias_df is not None:
        # analyzer 输出的格式是 occupation × province，需要转置为 province × occupation
        pivot_bias = pivot_bias_df.T.fillna(0)
        print(f"  使用预加载的 pivot 文件（余弦相似度差值）")
    else:
        # 从长格式数据创建 pivot
        pivot_bias = occupation_df.pivot_table(
            values="bias_score", index="province", columns="occupation", aggfunc="mean"
        ).fillna(0)
        print(f"  从长格式数据创建 pivot（余弦相似度差值）")

    # 检查数据是否足够进行聚类分析
    if len(pivot_bias) < 2:
        print(f"⚠️  省份数量不足（需要至少2个），跳过聚类分析")
        return

    # 同样处理 projection_score
    if pivot_projection_df is not None:
        pivot_projection = pivot_projection_df.T.fillna(0)
        print(f"  使用预加载的 pivot 文件（性别轴投影）")
    else:
        pivot_projection = occupation_df.pivot_table(
            values="projection_score",
            index="province",
            columns="occupation",
            aggfunc="mean",
        ).fillna(0)
        print(f"  从长格式数据创建 pivot（性别轴投影）")

    # 使用 bias_score 进行聚类分析
    pivot = pivot_bias

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

    # 使用 projection_score 进行聚类分析（如果数据可用）
    if pivot_projection is not None and len(pivot_projection) > 0:
        print(f"\n  生成性别轴投影方法的聚类分析...")

        # 层次聚类
        linkage_matrix_proj = linkage(pivot_projection, method="ward")

        # 绘制树状图
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))

        dendrogram(
            linkage_matrix_proj,
            labels=pivot_projection.index.tolist(),
            leaf_font_size=11,
            ax=ax,
            color_threshold=0.7 * max(linkage_matrix_proj[:, 2]),
        )

        ax.set_title(
            f"省份性别观念模式聚类分析 ({year}年)\n基于职业性别偏向模式（性别轴投影方法）",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        ax.set_xlabel("省份", fontsize=12, fontweight="bold")
        ax.set_ylabel("距离（差异程度）", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        plt.tight_layout()
        cluster_file_proj = os.path.join(
            OUTPUT_DIR, f"province_clustering_projection_{year}.pdf"
        )
        plt.savefig(cluster_file_proj, format="pdf", bbox_inches="tight")
        print(f"✓ 省份聚类图（投影方法）已保存: {cluster_file_proj}")
        plt.close()

        # 绘制热力图：省份相似度矩阵
        distances_proj = pdist(pivot_projection, metric="euclidean")
        distance_matrix_proj = squareform(distances_proj)

        # 转换为相似度
        max_dist_proj = distance_matrix_proj.max()
        similarity_matrix_proj = 1 - (distance_matrix_proj / max_dist_proj)

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        sns.heatmap(
            similarity_matrix_proj,
            xticklabels=pivot_projection.index,
            yticklabels=pivot_projection.index,
            annot=False,
            fmt=".2f",
            cmap="YlOrRd",
            cbar_kws={"label": "模式相似度"},
            ax=ax,
            square=True,
        )

        ax.set_title(
            f"省份性别观念模式相似度矩阵 ({year}年，性别轴投影方法)\n颜色越深 = 模式越相似",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        plt.tight_layout()
        similarity_file_proj = os.path.join(
            OUTPUT_DIR, f"province_similarity_projection_{year}.pdf"
        )
        plt.savefig(similarity_file_proj, format="pdf", bbox_inches="tight")
        print(f"✓ 省份相似度矩阵（投影方法）已保存: {similarity_file_proj}")
        plt.close()


def plot_province_comparison(stats_df, year):
    """省份多维度对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 隔离程度 vs 平均偏向
    ax = axes[0, 0]
    # 使用 vocab_size 作为点的大小（如果存在），否则使用固定大小
    size_col = "vocab_size" if "vocab_size" in stats_df.columns else None
    if size_col:
        sizes = stats_df[size_col] / 1000
    else:
        sizes = 100  # 固定大小

    scatter = ax.scatter(
        stats_df["mean_bias"],
        stats_df["std_bias"],
        s=sizes,
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
    # 使用 vocab_size 作为数据量指标（如果存在），否则使用 occupations_found
    if "vocab_size" in stats_df.columns:
        size_col = "vocab_size"
        unit = 1000
        ylabel = "词汇表大小（千）"
    elif "occupations_found" in stats_df.columns:
        size_col = "occupations_found"
        unit = 1
        ylabel = "找到的职业数"
    else:
        # 如果没有可用字段，跳过这个子图
        ax.text(
            0.5, 0.5, "无数据量信息", ha="center", va="center", transform=ax.transAxes
        )
        ax.set_title("各省份数据量分布", fontsize=12, fontweight="bold")
        ax.axis("off")
        size_col = None

    if size_col:
        stats_sorted = stats_df.sort_values(size_col, ascending=False)
        bars = ax.bar(
            range(len(stats_sorted)),
            stats_sorted[size_col] / unit,
            color="steelblue",
            alpha=0.7,
        )
        ax.set_xticks(range(len(stats_sorted)))
        ax.set_xticklabels(
            stats_sorted["province"], rotation=45, ha="right", fontsize=9
        )
        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
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
    """特定职业在各省份的性别偏向对比（包含两种方法）"""
    occ_data = occupation_df[occupation_df["occupation"] == occupation_name].copy()

    if len(occ_data) == 0:
        print(f"⚠️  未找到职业: {occupation_name}")
        return

    occ_data = occ_data.sort_values("bias_score")

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # 1. 左上：余弦相似度差值方法
    ax1 = axes[0, 0]
    colors = ["#d62728" if x < 0 else "#2ca02c" for x in occ_data["bias_score"]]
    bars = ax1.barh(
        occ_data["province"], occ_data["bias_score"], color=colors, alpha=0.7
    )

    ax1.axvline(x=0, color="black", linestyle="--", linewidth=1)
    ax1.set_xlabel(
        "性别偏向分数\n(负=偏男性, 正=偏女性)", fontsize=11, fontweight="bold"
    )
    ax1.set_title(
        f'"{occupation_name}"的性别关联（余弦相似度差值方法）', fontsize=12, fontweight="bold"
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

    # 2. 右上：性别轴投影方法
    ax2 = axes[0, 1]
    occ_data_proj = occ_data.sort_values("projection_score")
    colors_proj = ["#d62728" if x < 0 else "#2ca02c" for x in occ_data_proj["projection_score"]]
    bars_proj = ax2.barh(
        occ_data_proj["province"], occ_data_proj["projection_score"], color=colors_proj, alpha=0.7
    )

    ax2.axvline(x=0, color="black", linestyle="--", linewidth=1)
    ax2.set_xlabel(
        "性别轴投影分数\n(负=偏男性, 正=偏女性)", fontsize=11, fontweight="bold"
    )
    ax2.set_title(
        f'"{occupation_name}"的性别关联（性别轴投影方法）', fontsize=12, fontweight="bold"
    )
    ax2.grid(axis="x", alpha=0.3)

    for bar in bars_proj:
        width = bar.get_width()
        ax2.text(
            width + (0.005 if width > 0 else -0.005),
            bar.get_y() + bar.get_height() / 2,
            f"{width:+.3f}",
            va="center",
            ha="left" if width > 0 else "right",
            fontsize=8,
        )

    # 3. 左下：男性/女性相似度对比
    ax3 = axes[1, 0]
    x = np.arange(len(occ_data))
    width = 0.35

    bars1 = ax3.barh(
        x - width / 2,
        occ_data["male_similarity"],
        width,
        label="男性相似度",
        color="#1f77b4",
        alpha=0.7,
    )
    bars2 = ax3.barh(
        x + width / 2,
        occ_data["female_similarity"],
        width,
        label="女性相似度",
        color="#ff7f0e",
        alpha=0.7,
    )

    ax3.set_yticks(x)
    ax3.set_yticklabels(occ_data["province"])
    ax3.set_xlabel("与性别词的相似度", fontsize=11, fontweight="bold")
    ax3.set_title(
        f'"{occupation_name}"与性别词的相似度分解', fontsize=12, fontweight="bold"
    )
    ax3.legend(fontsize=10)
    ax3.grid(axis="x", alpha=0.3)

    # 4. 右下：两种方法的对比散点图
    ax4 = axes[1, 1]
    scatter = ax4.scatter(
        occ_data["bias_score"],
        occ_data["projection_score"],
        s=100,
        alpha=0.6,
        edgecolors="black",
        linewidth=1,
    )

    for _, row in occ_data.iterrows():
        ax4.annotate(
            row["province"],
            (row["bias_score"], row["projection_score"]),
            fontsize=8,
            ha="center",
        )

    ax4.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax4.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax4.set_xlabel("余弦相似度差值", fontsize=11, fontweight="bold")
    ax4.set_ylabel("性别轴投影分数", fontsize=11, fontweight="bold")
    ax4.set_title(
        f'"{occupation_name}"两种方法对比', fontsize=12, fontweight="bold"
    )
    ax4.grid(alpha=0.3)

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

        # 使用可用的数据量字段
        if "vocab_size" in stats_df.columns:
            f.write(f"  总词汇表大小: {stats_df['vocab_size'].sum():,}\n")
            f.write(f"  平均每省份词汇表大小: {stats_df['vocab_size'].mean():,.0f}\n")
            f.write(
                f"  词汇表最大的省份: {stats_df.nlargest(1, 'vocab_size')['province'].values[0]}\n"
            )
            f.write(
                f"  词汇表最小的省份: {stats_df.nsmallest(1, 'vocab_size')['province'].values[0]}\n"
            )
        elif "occupations_found" in stats_df.columns:
            f.write(f"  总找到职业数: {stats_df['occupations_found'].sum():,}\n")
            f.write(
                f"  平均每省份找到职业数: {stats_df['occupations_found'].mean():,.0f}\n"
            )
            f.write(
                f"  找到职业最多的省份: {stats_df.nlargest(1, 'occupations_found')['province'].values[0]}\n"
            )
            f.write(
                f"  找到职业最少的省份: {stats_df.nsmallest(1, 'occupations_found')['province'].values[0]}\n"
            )
        else:
            f.write(f"  (无数据量信息)\n")

    print(f"✓ 分析总结已保存: {report_file}")


def plot_occupation_overview(occupation_df, year):
    """职业性别偏向综合分析（两种方法对比）"""
    print(f"  生成职业性别偏向综合分析图...")

    # 计算每个职业的跨省份平均值
    occ_avg = occupation_df.groupby("occupation").agg({
        "bias_score": ["mean", "std"],
        "projection_score": ["mean", "std"]
    }).reset_index()
    occ_avg.columns = ["occupation", "bias_mean", "bias_std", "proj_mean", "proj_std"]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # 1. 左上：余弦相似度差值方法 - 最偏女性/男性职业
    ax1 = axes[0, 0]
    top_female = occ_avg.nlargest(10, "bias_mean").sort_values("bias_mean")
    top_male = occ_avg.nsmallest(10, "bias_mean").sort_values("bias_mean")
    combined = pd.concat([top_male, top_female])

    colors = ["#d62728" if x < 0 else "#2ca02c" for x in combined["bias_mean"]]
    bars = ax1.barh(combined["occupation"], combined["bias_mean"], color=colors, alpha=0.7)

    ax1.axvline(x=0, color="black", linestyle="--", linewidth=1)
    ax1.set_xlabel("平均性别偏向分数", fontsize=11, fontweight="bold")
    ax1.set_title("职业性别偏向排名（余弦相似度差值方法）", fontsize=12, fontweight="bold")
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

    # 2. 右上：性别轴投影方法 - 最偏女性/男性职业
    ax2 = axes[0, 1]
    top_female_proj = occ_avg.nlargest(10, "proj_mean").sort_values("proj_mean")
    top_male_proj = occ_avg.nsmallest(10, "proj_mean").sort_values("proj_mean")
    combined_proj = pd.concat([top_male_proj, top_female_proj])

    colors_proj = ["#d62728" if x < 0 else "#2ca02c" for x in combined_proj["proj_mean"]]
    bars_proj = ax2.barh(combined_proj["occupation"], combined_proj["proj_mean"],
                         color=colors_proj, alpha=0.7)

    ax2.axvline(x=0, color="black", linestyle="--", linewidth=1)
    ax2.set_xlabel("平均性别轴投影分数", fontsize=11, fontweight="bold")
    ax2.set_title("职业性别偏向排名（性别轴投影方法）", fontsize=12, fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)

    for bar in bars_proj:
        width = bar.get_width()
        ax2.text(
            width + (0.005 if width > 0 else -0.005),
            bar.get_y() + bar.get_height() / 2,
            f"{width:+.3f}",
            va="center",
            ha="left" if width > 0 else "right",
            fontsize=8,
        )

    # 3. 左下：两种方法相关性散点图
    ax3 = axes[1, 0]
    scatter = ax3.scatter(
        occ_avg["bias_mean"],
        occ_avg["proj_mean"],
        s=100,
        alpha=0.6,
        edgecolors="black",
        linewidth=1,
    )

    # 计算相关系数
    corr = occ_avg[["bias_mean", "proj_mean"]].corr().iloc[0, 1]

    # 添加部分职业标签
    for _, row in occ_avg.nlargest(5, "bias_mean").iterrows():
        ax3.annotate(row["occupation"], (row["bias_mean"], row["proj_mean"]),
                    fontsize=8, ha="center")
    for _, row in occ_avg.nsmallest(5, "bias_mean").iterrows():
        ax3.annotate(row["occupation"], (row["bias_mean"], row["proj_mean"]),
                    fontsize=8, ha="center")

    ax3.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax3.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax3.set_xlabel("余弦相似度差值", fontsize=11, fontweight="bold")
    ax3.set_ylabel("性别轴投影分数", fontsize=11, fontweight="bold")
    ax3.set_title(f"两种方法对比 (相关系数: {corr:.3f})", fontsize=12, fontweight="bold")
    ax3.grid(alpha=0.3)

    # 4. 右下：职业间差异程度对比
    ax4 = axes[1, 1]
    occ_variability = occ_avg.nlargest(15, "bias_std").sort_values("bias_std")

    x = np.arange(len(occ_variability))
    width = 0.35

    bars1 = ax4.barh(x - width/2, occ_variability["bias_std"], width,
                     label="余弦相似度差值", color="#1f77b4", alpha=0.7)
    bars2 = ax4.barh(x + width/2, occ_variability["proj_std"], width,
                     label="性别轴投影", color="#ff7f0e", alpha=0.7)

    ax4.set_yticks(x)
    ax4.set_yticklabels(occ_variability["occupation"])
    ax4.set_xlabel("省份间差异（标准差）", fontsize=11, fontweight="bold")
    ax4.set_title("职业性别观念省份间差异最大的职业", fontsize=12, fontweight="bold")
    ax4.legend(fontsize=10)
    ax4.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    overview_file = os.path.join(OUTPUT_DIR, f"occupation_overview_{year}.pdf")
    plt.savefig(overview_file, format="pdf", bbox_inches="tight")
    print(f"✓ 职业综合分析图已保存: {overview_file}")
    plt.close()


def plot_domestic_work_comprehensive(occupation_df, year):
    """家务分工场域分析（两种方法综合）

    从occupation_df加载domestic_work_bias.csv数据进行分析
    """
    print(f"  生成家务分工场域综合分析图...")

    # 加载家务分工数据
    year_input_dir = os.path.join(INPUT_DIR, str(year))
    domestic_work_file = os.path.join(year_input_dir, "domestic_work_bias.csv")
    domestic_work_projection_file = os.path.join(
        year_input_dir, "domestic_work_gender_projection.csv"
    )

    if not os.path.exists(domestic_work_file):
        print(f"  ⚠️  未找到家务分工数据: {domestic_work_file}")
        return

    domestic_work_df = pd.read_csv(domestic_work_file)

    # 转换省份编码
    def convert_province_code(province):
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

    domestic_work_df["province"] = domestic_work_df["province"].apply(convert_province_code)
    domestic_work_df = domestic_work_df[domestic_work_df["province"] != "未知"].copy()

    # 加载投影数据（如果存在）
    domestic_work_projection_df = None
    if os.path.exists(domestic_work_projection_file):
        domestic_work_projection_df = pd.read_csv(domestic_work_projection_file)
        domestic_work_projection_df["province"] = domestic_work_projection_df["province"].apply(
            convert_province_code
        )
        domestic_work_projection_df = domestic_work_projection_df[
            domestic_work_projection_df["province"] != "未知"
        ].copy()

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # 1. 左上：余弦相似度差值方法 - 性别在家庭/工作场域的偏向
    ax1 = axes[0, 0]

    # 计算每个省份的性别场域差异
    province_gaps = []
    for province in domestic_work_df["province"].unique():
        prov_df = domestic_work_df[domestic_work_df["province"] == province]
        male_bias = prov_df[prov_df["word_type"] == "male"]["domain_bias"].values
        female_bias = prov_df[prov_df["word_type"] == "female"]["domain_bias"].values

        if len(male_bias) > 0 and len(female_bias) > 0:
            gap = female_bias[0] - male_bias[0]
            province_gaps.append({
                "province": province,
                "male_domain_bias": male_bias[0],
                "female_domain_bias": female_bias[0],
                "gender_domain_gap": gap
            })

    gaps_df = pd.DataFrame(province_gaps).sort_values("gender_domain_gap")

    colors = ["#d62728" if x < 0 else "#2ca02c" for x in gaps_df["gender_domain_gap"]]
    bars = ax1.barh(gaps_df["province"], gaps_df["gender_domain_gap"],
                    color=colors, alpha=0.7)

    ax1.axvline(x=0, color="black", linestyle="--", linewidth=1)
    ax1.set_xlabel("性别场域差异\n(正=女性更偏家庭)", fontsize=11, fontweight="bold")
    ax1.set_title("性别在家庭-工作场域的差异（余弦相似度差值方法）",
                  fontsize=12, fontweight="bold")
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

    # 2. 右上：性别轴投影方法 - 场域性别偏向
    ax2 = axes[0, 1]
    if domestic_work_projection_df is not None and len(domestic_work_projection_df) > 0:
        proj_sorted = domestic_work_projection_df.sort_values("domain_gender_bias")
        colors_proj = ["#d62728" if x < 0 else "#2ca02c"
                      for x in proj_sorted["domain_gender_bias"]]
        bars_proj = ax2.barh(proj_sorted["province"],
                            proj_sorted["domain_gender_bias"],
                            color=colors_proj, alpha=0.7)

        ax2.axvline(x=0, color="black", linestyle="--", linewidth=1)
        ax2.set_xlabel("场域性别偏向\n(正=family更偏女性)", fontsize=11, fontweight="bold")
        ax2.set_title("家庭-工作场域性别偏向（性别轴投影方法）",
                      fontsize=12, fontweight="bold")
        ax2.grid(axis="x", alpha=0.3)

        for bar in bars_proj:
            width = bar.get_width()
            ax2.text(
                width + (0.005 if width > 0 else -0.005),
                bar.get_y() + bar.get_height() / 2,
                f"{width:+.3f}",
                va="center",
                ha="left" if width > 0 else "right",
                fontsize=8,
            )
    else:
        ax2.text(0.5, 0.5, "无投影数据", ha="center", va="center",
                transform=ax2.transAxes, fontsize=14)
        ax2.set_title("家庭-工作场域性别偏向（性别轴投影方法）",
                      fontsize=12, fontweight="bold")

    # 3. 左下：男性/女性在家庭-工作场域的偏向对比（余弦相似度）
    ax3 = axes[1, 0]
    x = np.arange(len(gaps_df))
    width = 0.35

    bars1 = ax3.barh(x - width/2, gaps_df["male_domain_bias"], width,
                     label="男性场域偏向", color="#1f77b4", alpha=0.7)
    bars2 = ax3.barh(x + width/2, gaps_df["female_domain_bias"], width,
                     label="女性场域偏向", color="#ff7f0e", alpha=0.7)

    ax3.set_yticks(x)
    ax3.set_yticklabels(gaps_df["province"])
    ax3.axvline(x=0, color="black", linestyle="--", linewidth=1)
    ax3.set_xlabel("场域偏向分数\n(正=偏家庭, 负=偏工作)", fontsize=11, fontweight="bold")
    ax3.set_title("男性/女性在家庭-工作场域的偏向对比", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=10)
    ax3.grid(axis="x", alpha=0.3)

    # 4. 右下：两种方法的相关性分析
    ax4 = axes[1, 1]
    if domestic_work_projection_df is not None and len(domestic_work_projection_df) > 0:
        # 合并两个数据框
        merged = gaps_df.merge(domestic_work_projection_df, on="province")

        scatter = ax4.scatter(
            merged["gender_domain_gap"],
            merged["domain_gender_bias"],
            s=100,
            alpha=0.6,
            edgecolors="black",
            linewidth=1,
        )

        for _, row in merged.iterrows():
            ax4.annotate(
                row["province"],
                (row["gender_domain_gap"], row["domain_gender_bias"]),
                fontsize=8,
                ha="center",
            )

        # 计算相关系数
        corr = merged[["gender_domain_gap", "domain_gender_bias"]].corr().iloc[0, 1]

        ax4.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax4.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
        ax4.set_xlabel("余弦相似度差值方法", fontsize=11, fontweight="bold")
        ax4.set_ylabel("性别轴投影方法", fontsize=11, fontweight="bold")
        ax4.set_title(f"两种方法对比 (相关系数: {corr:.3f})", fontsize=12, fontweight="bold")
        ax4.grid(alpha=0.3)
    else:
        ax4.text(0.5, 0.5, "无投影数据", ha="center", va="center",
                transform=ax4.transAxes, fontsize=14)
        ax4.set_title("两种方法对比", fontsize=12, fontweight="bold")

    plt.tight_layout()
    comprehensive_file = os.path.join(OUTPUT_DIR,
                                     f"domestic_work_comprehensive_{year}.pdf")
    plt.savefig(comprehensive_file, format="pdf", bbox_inches="tight")
    print(f"✓ 家务分工综合分析图已保存: {comprehensive_file}")
    plt.close()


def plot_domestic_work_gender_projection(domestic_work_projection_df, year):
    """绘制家务分工词汇在性别轴上的投影分析"""
    if domestic_work_projection_df is None or len(domestic_work_projection_df) == 0:
        print("⚠️  没有家务分工投影数据，跳过可视化")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 各省份的work和family词汇平均投影对比
    ax = axes[0, 0]
    df_sorted = domestic_work_projection_df.sort_values("domain_gender_bias")
    x = np.arange(len(df_sorted))
    width = 0.35

    bars1 = ax.barh(
        x - width / 2,
        df_sorted["work_mean_projection"],
        width,
        label="work词汇平均投影",
        color="#1f77b4",
        alpha=0.7,
    )
    bars2 = ax.barh(
        x + width / 2,
        df_sorted["family_mean_projection"],
        width,
        label="family词汇平均投影",
        color="#ff7f0e",
        alpha=0.7,
    )

    ax.set_yticks(x)
    ax.set_yticklabels(df_sorted["province"])
    ax.set_xlabel("在性别轴上的投影分数", fontsize=11, fontweight="bold")
    ax.set_title(
        "各省份work/family词汇在性别轴上的投影对比", fontsize=12, fontweight="bold"
    )
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.legend(fontsize=10)
    ax.grid(axis="x", alpha=0.3)

    # 2. 场域性别偏向（domain_gender_bias）排名
    ax = axes[0, 1]
    df_sorted_bias = domestic_work_projection_df.sort_values(
        "domain_gender_bias", ascending=True
    )
    colors = [
        "#d62728" if x < 0 else "#2ca02c" for x in df_sorted_bias["domain_gender_bias"]
    ]
    bars = ax.barh(
        df_sorted_bias["province"],
        df_sorted_bias["domain_gender_bias"],
        color=colors,
        alpha=0.7,
    )

    ax.axvline(x=0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(
        "场域性别偏向分数\n(负=work更偏女性, 正=family更偏女性)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_title("各省份场域性别偏向排名", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + (0.01 if width > 0 else -0.01),
            bar.get_y() + bar.get_height() / 2,
            f"{width:+.3f}",
            va="center",
            ha="left" if width > 0 else "right",
            fontsize=8,
        )

    # 3. work和family投影的散点图
    ax = axes[1, 0]
    scatter = ax.scatter(
        domestic_work_projection_df["work_mean_projection"],
        domestic_work_projection_df["family_mean_projection"],
        s=100,
        c=domestic_work_projection_df["domain_gender_bias"],
        cmap="RdYlGn",
        alpha=0.6,
        edgecolors="black",
        linewidth=1,
    )

    for _, row in domestic_work_projection_df.iterrows():
        ax.annotate(
            row["province"],
            (row["work_mean_projection"], row["family_mean_projection"]),
            fontsize=8,
            ha="center",
        )

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.plot([-1, 1], [-1, 1], "k--", alpha=0.3, label="y=x")

    ax.set_xlabel(
        "work词汇平均投影\n(负=偏男性, 正=偏女性)", fontsize=11, fontweight="bold"
    )
    ax.set_ylabel(
        "family词汇平均投影\n(负=偏男性, 正=偏女性)", fontsize=11, fontweight="bold"
    )
    ax.set_title(
        "work vs family词汇在性别轴上的投影关系", fontsize=12, fontweight="bold"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    plt.colorbar(scatter, ax=ax, label="场域性别偏向")

    # 4. 按区域分组的箱线图
    ax = axes[1, 1]
    # 添加区域信息（创建副本避免修改原数据）
    if "region" not in domestic_work_projection_df.columns:
        df_with_region = domestic_work_projection_df.copy()
        df_with_region["region"] = df_with_region["province"].map(PROVINCE_TO_REGION)
        df_with_region["region"] = df_with_region["region"].fillna("未知区域")
    else:
        df_with_region = domestic_work_projection_df

    region_order = ["华北", "东北", "华东", "华中", "华南", "西南", "西北"]
    data_by_region = [
        df_with_region[df_with_region["region"] == r]["domain_gender_bias"].values
        for r in region_order
        if r in df_with_region["region"].values
    ]
    labels_with_data = [r for r in region_order if r in df_with_region["region"].values]

    if data_by_region:
        bp = ax.boxplot(data_by_region, labels=labels_with_data, patch_artist=True)
        for patch, color in zip(
            bp["boxes"], plt.cm.Set3(np.linspace(0, 1, len(data_by_region)))
        ):
            patch.set_facecolor(color)

        ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_ylabel("场域性别偏向分数", fontsize=11, fontweight="bold")
        ax.set_title("各地理区域场域性别偏向分布", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    projection_file = os.path.join(
        OUTPUT_DIR, f"domestic_work_gender_projection_{year}.pdf"
    )
    plt.savefig(projection_file, format="pdf", bbox_inches="tight")
    print(f"✓ 家务分工词汇性别轴投影分析图已保存: {projection_file}")
    plt.close()


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
    (
        stats_df,
        occupation_df,
        pivot_bias_df,
        pivot_projection_df,
        domestic_work_projection_df,
    ) = load_results(year)
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
    plot_province_clustering(
        occupation_df, stats_df, year, pivot_bias_df, pivot_projection_df
    )

    # 4. 省份多维度对比
    print(f"\n📈 生成省份对比图...")
    plot_province_comparison(stats_df, year)

    # 5. **NEW** 职业性别偏向综合分析（两种方法）
    print(f"\n💼 生成职业性别偏向综合分析...")
    plot_occupation_overview(occupation_df, year)

    # 6. 特定职业的省份差异（更新为包含两种方法）
    print(f"\n👔 生成典型职业分析...")
    key_occupations = ["护士", "程序员", "教师", "医生", "CEO"]
    for occ in key_occupations:
        if occ in occupation_df["occupation"].values:
            plot_occupation_by_province(occupation_df, occ, year)

    # 7. **NEW** 家务分工场域综合分析（两种方法）
    print(f"\n🏠 生成家务分工场域综合分析...")
    plot_domestic_work_comprehensive(occupation_df, year)

    # 8. 家务分工词汇性别轴投影分析（原有功能，保留）
    print(f"\n🏠 生成家务分工词汇性别轴投影详细分析...")
    plot_domestic_work_gender_projection(domestic_work_projection_df, year)

    # 9. 生成总结报告
    print(f"\n📝 生成总结报告...")
    generate_summary_report(stats_df, occupation_df, year)

    print(f"\n{'='*70}")
    print(f"✅ 可视化完成！所有文件已保存到: {OUTPUT_DIR}/")
    print(f"{'='*70}\n")

    print(f"生成的文件包括:")
    print(f"\n【省份层面分析】")
    print(f"  1. segregation_map_{year}.pdf - 中国地图（性别隔离程度）")
    print(f"  2. segregation_map_regional_{year}.pdf - 中国地图（按区域着色）")
    print(f"  3. segregation_ranking_{year}.pdf - 省份排名图")
    print(f"  4. province_clustering_{year}.pdf - 省份聚类树状图（余弦相似度差值方法）")
    print(f"  5. province_similarity_{year}.pdf - 省份相似度热力图（余弦相似度差值方法）")
    print(f"  6. province_clustering_projection_{year}.pdf - 省份聚类树状图（性别轴投影方法）")
    print(f"  7. province_similarity_projection_{year}.pdf - 省份相似度热力图（性别轴投影方法）")
    print(f"  8. province_comparison_{year}.pdf - 省份多维度对比")
    print(f"\n【职业性别偏向分析】")
    print(f"  9. occupation_overview_{year}.pdf - **NEW** 职业性别偏向综合分析（两种方法对比）")
    print(f"  10. occupation_[职业名]_{year}.pdf - 特定职业的省份分析（更新为包含两种方法）")
    print(f"\n【家务分工场域分析】")
    print(f"  11. domestic_work_comprehensive_{year}.pdf - **NEW** 家务分工场域综合分析（两种方法对比）")
    print(f"  12. domestic_work_gender_projection_{year}.pdf - 家务分工词汇性别轴投影详细分析")
    print(f"\n【总结报告】")
    print(f"  13. visualization_summary_{year}.txt - 文字总结报告\n")
    print(f"\n✨ **主要更新**：")
    print(f"  - 所有分析现在包含两种方法：余弦相似度差值 + 性别轴投影")
    print(f"  - 新增职业性别偏向综合分析，直接对比两种方法")
    print(f"  - 新增家务分工场域综合分析，展示两种方法的一致性")
    print(f"  - 更新职业分析图，包含4个子图展示完整信息\n")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
