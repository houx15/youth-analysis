"""
省级Gender Norm Index构建器 (报纸数据版)

基于报纸语料库的性别规范指数构建:
- Step 0: OOV检查
- Step 1: 构建各省性别轴
- Step 2: 计算各概念词在性别轴上的投影
- Step 3: 检查跨省可比性
- Step 4: 计算WEAT效应量(Cohen's d)
- Step 5: 输出结果

输入：newspaper_data/embedding_models/ 下的模型文件
输出：newspaper_data/gender_norm_index/ 下的分析结果
"""

import os
import json
import glob
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
import fire

warnings.filterwarnings("ignore")


# =============================================================================
# 配置
# =============================================================================

MODEL_DIR = "gender_norms/newspaper_data/embedding_models"
OUTPUT_DIR = "gender_norms/newspaper_data/gender_norm_index"
WORDLISTS_DIR = "gender_norms/wordlists"

DATA_YEAR = 2019

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# 数据类
# =============================================================================


@dataclass
class WordlistConfig:
    """词表配置"""

    gender_file: str = "gender_words.json"
    domestic_work_file: str = "domestic_work_words.json"
    leadership_file: str = "leadership_words.json"
    stem_file: str = "stem_words.json"


@dataclass
class Wordlists:
    """所有词表"""

    male: List[str] = field(default_factory=list)
    female: List[str] = field(default_factory=list)
    family: List[str] = field(default_factory=list)
    work: List[str] = field(default_factory=list)
    leadership: List[str] = field(default_factory=list)
    non_leadership: List[str] = field(default_factory=list)
    stem: List[str] = field(default_factory=list)
    non_stem: List[str] = field(default_factory=list)

    def get_all_concept_words(self) -> Dict[str, List[str]]:
        """获取所有概念词(不含性别词)"""
        return {
            "family": self.family,
            "work": self.work,
            "leadership": self.leadership,
            "non_leadership": self.non_leadership,
            "stem": self.stem,
            "non_stem": self.non_stem,
        }

    def get_all_categories(self) -> Dict[str, List[str]]:
        """获取所有词表类别"""
        return {
            "male": self.male,
            "female": self.female,
            **self.get_all_concept_words(),
        }


@dataclass
class OOVReport:
    """OOV报告"""

    province: str
    category: str
    total_words: int
    found_words: int
    oov_words: List[str]

    @property
    def coverage_rate(self) -> float:
        if self.total_words == 0:
            return 0.0
        return self.found_words / self.total_words


@dataclass
class ProvinceGenderAxis:
    """省份性别轴"""

    province: str
    male_centroid: np.ndarray
    female_centroid: np.ndarray
    gender_axis: np.ndarray
    male_words_found: List[str]
    female_words_found: List[str]


@dataclass
class WordProjection:
    """词的投影结果"""

    province: str
    word: str
    category: str
    projection: float
    cosine_sim: float


@dataclass
class ProvinceStats:
    """省份统计信息"""

    province: str
    mean_projection: float
    std_projection: float
    min_projection: float
    max_projection: float


@dataclass
class WEATResult:
    """WEAT效应量结果"""

    province: str
    dimension: str
    cohens_d: float
    group1_mean: float
    group2_mean: float
    group1_std: float
    group2_std: float
    group1_n: int
    group2_n: int
    pooled_std: float


# =============================================================================
# 词表加载
# =============================================================================


def load_json_wordlist(filename: str) -> dict:
    """从JSON文件加载词表"""
    filepath = os.path.join(WORDLISTS_DIR, filename)
    if not os.path.exists(filepath):
        print(f"  [警告] 词表文件不存在: {filepath}")
        return {}

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  [错误] 加载词表文件失败 {filepath}: {e}")
        return {}


def load_all_wordlists(config: WordlistConfig = None) -> Wordlists:
    """加载所有词表"""
    if config is None:
        config = WordlistConfig()

    gender_data = load_json_wordlist(config.gender_file)
    male_words = gender_data.get("male", [])
    female_words = gender_data.get("female", [])

    domestic_data = load_json_wordlist(config.domestic_work_file)
    family_words = domestic_data.get("family", [])
    work_words = domestic_data.get("work", [])

    leadership_data = load_json_wordlist(config.leadership_file)
    leadership_words = leadership_data.get("leadership", [])
    non_leadership_words = leadership_data.get("non_leadership", [])

    stem_data = load_json_wordlist(config.stem_file)
    stem_words = stem_data.get("stem", [])
    non_stem_words = stem_data.get("non_stem", [])

    wordlists = Wordlists(
        male=male_words,
        female=female_words,
        family=family_words,
        work=work_words,
        leadership=leadership_words,
        non_leadership=non_leadership_words,
        stem=stem_words,
        non_stem=non_stem_words,
    )

    print(f"词表加载完成:")
    print(f"  - 男性词: {len(wordlists.male)} 个")
    print(f"  - 女性词: {len(wordlists.female)} 个")
    print(f"  - 家庭场域词: {len(wordlists.family)} 个")
    print(f"  - 工作场域词: {len(wordlists.work)} 个")
    print(f"  - 领导力词: {len(wordlists.leadership)} 个")
    print(f"  - 非领导力词: {len(wordlists.non_leadership)} 个")
    print(f"  - STEM词: {len(wordlists.stem)} 个")
    print(f"  - 非STEM词: {len(wordlists.non_stem)} 个")

    return wordlists


# =============================================================================
# 模型加载
# =============================================================================


def get_available_provinces() -> List[str]:
    """获取所有可用的省份列表"""
    pattern = os.path.join(MODEL_DIR, "model_*.kv")
    model_files = sorted(glob.glob(pattern))

    provinces = []
    for model_path in model_files:
        filename = os.path.basename(model_path)
        province = filename.replace("model_", "").replace(".kv", "")
        provinces.append(province)

    return sorted(provinces)


def load_model(province: str) -> Optional[KeyedVectors]:
    """加载指定省份的模型"""
    model_path = os.path.join(MODEL_DIR, f"model_{province}.kv")

    if not os.path.exists(model_path):
        return None

    try:
        model = KeyedVectors.load(model_path)
        return model
    except Exception as e:
        print(f"  [错误] 加载模型失败 {province}: {e}")
        return None


# =============================================================================
# Step 0: OOV检查
# =============================================================================


def check_word_in_vocab(model: KeyedVectors, word: str) -> bool:
    """检查词是否在词表中"""
    try:
        _ = model[word]
        return True
    except KeyError:
        return False


def check_oov_for_province(
    model: KeyedVectors, province: str, wordlists: Wordlists
) -> List[OOVReport]:
    """检查单个省份的OOV情况"""
    reports = []
    categories = wordlists.get_all_categories()

    for category, words in categories.items():
        found_words = []
        oov_words = []

        for word in words:
            if check_word_in_vocab(model, word):
                found_words.append(word)
            else:
                oov_words.append(word)

        report = OOVReport(
            province=province,
            category=category,
            total_words=len(words),
            found_words=len(found_words),
            oov_words=oov_words,
        )
        reports.append(report)

    return reports


def run_oov_check(
    wordlists: Wordlists,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
    """
    执行OOV检查

    Returns:
        province_coverage_df: 各省各词表覆盖率表
        word_coverage_df: 各词的跨省覆盖情况表
        province_oov_details: 各省的OOV详情
    """
    print(f"\n{'='*70}")
    print(f"Step 0: OOV检查")
    print(f"{'='*70}\n")

    provinces = get_available_provinces()
    if not provinces:
        print(f"  [错误] 未找到模型文件")
        return None, None, None

    print(f"  找到 {len(provinces)} 个省份模型")

    all_reports = []
    province_oov_details = {}

    word_province_found = {}
    categories = wordlists.get_all_categories()
    for category, words in categories.items():
        for word in words:
            word_province_found[word] = {"category": category, "found_in": []}

    for province in provinces:
        print(f"  检查省份: {province}")
        model = load_model(province)
        if model is None:
            print(f"    [跳过] 模型加载失败")
            continue

        reports = check_oov_for_province(model, province, wordlists)
        all_reports.extend(reports)

        province_oov_details[province] = {r.category: r.oov_words for r in reports}

        for category, words in categories.items():
            for word in words:
                if check_word_in_vocab(model, word):
                    word_province_found[word]["found_in"].append(province)

        del model

    province_coverage_data = []
    for report in all_reports:
        province_coverage_data.append(
            {
                "province": report.province,
                "category": report.category,
                "total": report.total_words,
                "found": report.found_words,
                "coverage": f"{report.coverage_rate:.1%}",
                "coverage_rate": report.coverage_rate,
            }
        )

    province_coverage_df = pd.DataFrame(province_coverage_data)

    word_coverage_data = []
    total_provinces = len(provinces)
    for word, info in word_province_found.items():
        n_found = len(info["found_in"])
        missing = [p for p in provinces if p not in info["found_in"]]
        word_coverage_data.append(
            {
                "word": word,
                "category": info["category"],
                "n_provinces_found": n_found,
                "coverage_rate": (
                    n_found / total_provinces if total_provinces > 0 else 0
                ),
                "missing_provinces": ",".join(missing[:5])
                + ("..." if len(missing) > 5 else ""),
            }
        )

    word_coverage_df = pd.DataFrame(word_coverage_data)
    word_coverage_df = word_coverage_df.sort_values(
        ["category", "coverage_rate"], ascending=[True, True]
    )

    print(f"\n  OOV检查摘要:")
    for category in categories.keys():
        cat_df = province_coverage_df[province_coverage_df["category"] == category]
        avg_coverage = cat_df["coverage_rate"].mean()
        print(f"    - {category}: 平均覆盖率 {avg_coverage:.1%}")

    high_oov_words = word_coverage_df[word_coverage_df["coverage_rate"] < 0.5]
    if len(high_oov_words) > 0:
        print(f"\n  [警告] 以下词在超过50%省份OOV(建议检查):")
        for _, row in high_oov_words.iterrows():
            print(
                f"    - [{row['category']}] {row['word']}: {row['n_provinces_found']}/{total_provinces}"
            )

    return province_coverage_df, word_coverage_df, province_oov_details


# =============================================================================
# Step 1: 构建性别轴
# =============================================================================


def get_word_vector(model: KeyedVectors, word: str) -> Optional[np.ndarray]:
    """获取词向量"""
    try:
        return model[word]
    except KeyError:
        return None


def compute_centroid(
    model: KeyedVectors, words: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """计算一组词的质心向量"""
    vectors = []
    found_words = []

    for word in words:
        vec = get_word_vector(model, word)
        if vec is not None:
            vectors.append(vec)
            found_words.append(word)

    if not vectors:
        return None, []

    centroid = np.mean(vectors, axis=0)
    return centroid, found_words


def build_gender_axis(
    model: KeyedVectors, province: str, wordlists: Wordlists
) -> Optional[ProvinceGenderAxis]:
    """
    构建省份的性别轴

    性别轴 = (女性质心 - 男性质心) / ||女性质心 - 男性质心||
    正方向指向女性
    """
    male_centroid, male_found = compute_centroid(model, wordlists.male)
    if male_centroid is None:
        print(f"    [警告] {province}: 未找到任何男性词")
        return None

    female_centroid, female_found = compute_centroid(model, wordlists.female)
    if female_centroid is None:
        print(f"    [警告] {province}: 未找到任何女性词")
        return None

    gender_axis = female_centroid - male_centroid
    axis_norm = np.linalg.norm(gender_axis)

    if axis_norm < 1e-10:
        print(f"    [警告] {province}: 性别轴长度接近0")
        return None

    gender_axis_normalized = gender_axis / axis_norm

    return ProvinceGenderAxis(
        province=province,
        male_centroid=male_centroid,
        female_centroid=female_centroid,
        gender_axis=gender_axis_normalized,
        male_words_found=male_found,
        female_words_found=female_found,
    )


def run_build_gender_axes(
    wordlists: Wordlists,
) -> Dict[str, ProvinceGenderAxis]:
    """为所有省份构建性别轴"""
    print(f"\n{'='*70}")
    print(f"Step 1: 构建各省性别轴")
    print(f"{'='*70}\n")

    provinces = get_available_provinces()
    gender_axes = {}

    for province in provinces:
        print(f"  处理省份: {province}")
        model = load_model(province)
        if model is None:
            continue

        axis = build_gender_axis(model, province, wordlists)
        if axis is not None:
            gender_axes[province] = axis
            print(
                f"    - 男性词: {len(axis.male_words_found)}/{len(wordlists.male)}, "
                f"女性词: {len(axis.female_words_found)}/{len(wordlists.female)}"
            )

        del model

    print(f"\n  成功构建 {len(gender_axes)} 个省份的性别轴")
    return gender_axes


# =============================================================================
# Step 2: 计算投影
# =============================================================================


def compute_projection(
    word_vec: np.ndarray, gender_axis: np.ndarray
) -> Tuple[float, float]:
    """
    计算词向量在性别轴上的投影

    Returns:
        projection: 点积投影值
        cosine_sim: 余弦相似度
    """
    projection = np.dot(word_vec, gender_axis)

    word_norm = np.linalg.norm(word_vec)
    if word_norm < 1e-10:
        cosine_sim = 0.0
    else:
        cosine_sim = projection / word_norm

    return float(projection), float(cosine_sim)


def compute_all_projections(
    wordlists: Wordlists, gender_axes: Dict[str, ProvinceGenderAxis]
) -> List[WordProjection]:
    """计算所有概念词在各省性别轴上的投影"""
    print(f"\n{'='*70}")
    print(f"Step 2: 计算各概念词在性别轴上的投影")
    print(f"{'='*70}\n")

    concept_words = wordlists.get_all_concept_words()
    all_projections = []

    for province, axis in gender_axes.items():
        print(f"  处理省份: {province}")
        model = load_model(province)
        if model is None:
            continue

        for category, words in concept_words.items():
            for word in words:
                word_vec = get_word_vector(model, word)
                if word_vec is not None:
                    projection, cosine_sim = compute_projection(
                        word_vec, axis.gender_axis
                    )
                    all_projections.append(
                        WordProjection(
                            province=province,
                            word=word,
                            category=category,
                            projection=projection,
                            cosine_sim=cosine_sim,
                        )
                    )

        del model

    print(f"\n  计算了 {len(all_projections)} 条投影记录")
    return all_projections


# =============================================================================
# Step 3: 检查跨省可比性
# =============================================================================


def compute_province_stats(projections: List[WordProjection]) -> List[ProvinceStats]:
    """计算每个省份的投影统计信息"""
    df = pd.DataFrame([vars(p) for p in projections])

    stats_list = []
    for province in df["province"].unique():
        prov_df = df[df["province"] == province]

        values = prov_df["cosine_sim"].values

        stats = ProvinceStats(
            province=province,
            mean_projection=float(np.mean(values)),
            std_projection=float(np.std(values)),
            min_projection=float(np.min(values)),
            max_projection=float(np.max(values)),
        )
        stats_list.append(stats)

    return stats_list


def run_comparability_check(
    projections: List[WordProjection],
) -> Tuple[List[ProvinceStats], bool]:
    """
    检查跨省可比性

    Returns:
        stats_list: 各省统计信息
        needs_standardization: 是否需要省内标准化
    """
    print(f"\n{'='*70}")
    print(f"Step 3: 检查跨省可比性")
    print(f"{'='*70}\n")

    stats_list = compute_province_stats(projections)

    means = [s.mean_projection for s in stats_list]
    stds = [s.std_projection for s in stats_list]

    mean_of_means = np.mean(means)
    std_of_means = np.std(means)
    mean_of_stds = np.mean(stds)
    std_of_stds = np.std(stds)

    print(f"  跨省投影值统计:")
    print(f"    - 各省均值的均值: {mean_of_means:.4f}")
    print(f"    - 各省均值的标准差: {std_of_means:.4f}")
    print(f"    - 各省标准差的均值: {mean_of_stds:.4f}")
    print(f"    - 各省标准差的标准差: {std_of_stds:.4f}")

    cv_means = std_of_means / abs(mean_of_means) if abs(mean_of_means) > 1e-10 else 0
    cv_stds = std_of_stds / mean_of_stds if mean_of_stds > 1e-10 else 0

    print(f"    - 均值变异系数 (CV): {cv_means:.4f}")
    print(f"    - 标准差变异系数 (CV): {cv_stds:.4f}")

    THRESHOLD = 0.3
    needs_standardization = cv_means > THRESHOLD or cv_stds > THRESHOLD

    if needs_standardization:
        print(f"\n  [建议] CV > {THRESHOLD}, 建议进行省内标准化")
    else:
        print(f"\n  [建议] CV <= {THRESHOLD}, 可不进行省内标准化")

    return stats_list, needs_standardization


def standardize_projections(
    projections: List[WordProjection], stats_list: List[ProvinceStats]
) -> pd.DataFrame:
    """对投影值进行省内z-score标准化"""
    df = pd.DataFrame([vars(p) for p in projections])

    stats_map = {s.province: s for s in stats_list}

    def compute_zscore(row):
        stats = stats_map.get(row["province"])
        if stats is None or stats.std_projection < 1e-10:
            return 0.0
        return (row["cosine_sim"] - stats.mean_projection) / stats.std_projection

    df["projection_zscore"] = df.apply(compute_zscore, axis=1)

    return df


# =============================================================================
# Step 4: 计算WEAT效应量
# =============================================================================


def compute_pooled_std(values1: np.ndarray, values2: np.ndarray) -> float:
    """计算pooled标准差"""
    n1 = len(values1)
    n2 = len(values2)

    if n1 < 2 or n2 < 2:
        return 0.0

    s1 = np.std(values1, ddof=1)
    s2 = np.std(values2, ddof=1)

    pooled_var = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)
    return np.sqrt(pooled_var)


def compute_cohens_d(
    group1_values: np.ndarray, group2_values: np.ndarray
) -> Tuple[float, float]:
    """
    计算Cohen's d效应量

    d = (mean1 - mean2) / pooled_std

    Returns:
        cohens_d: 效应量
        pooled_std: pooled标准差
    """
    mean1 = np.mean(group1_values)
    mean2 = np.mean(group2_values)

    pooled_std = compute_pooled_std(group1_values, group2_values)

    if pooled_std < 1e-10:
        return 0.0, 0.0

    cohens_d = (mean1 - mean2) / pooled_std
    return float(cohens_d), float(pooled_std)


def compute_weat_for_dimension(
    projection_df: pd.DataFrame,
    province: str,
    group1_category: str,
    group2_category: str,
    dimension_name: str,
    use_zscore: bool = False,
) -> Optional[WEATResult]:
    """计算单个维度的WEAT效应量"""
    prov_df = projection_df[projection_df["province"] == province]

    value_col = "projection_zscore" if use_zscore else "cosine_sim"

    group1_df = prov_df[prov_df["category"] == group1_category]
    group2_df = prov_df[prov_df["category"] == group2_category]

    if len(group1_df) == 0 or len(group2_df) == 0:
        return None

    group1_values = group1_df[value_col].values
    group2_values = group2_df[value_col].values

    cohens_d, pooled_std = compute_cohens_d(group1_values, group2_values)

    return WEATResult(
        province=province,
        dimension=dimension_name,
        cohens_d=cohens_d,
        group1_mean=float(np.mean(group1_values)),
        group2_mean=float(np.mean(group2_values)),
        group1_std=(
            float(np.std(group1_values, ddof=1)) if len(group1_values) > 1 else 0.0
        ),
        group2_std=(
            float(np.std(group2_values, ddof=1)) if len(group2_values) > 1 else 0.0
        ),
        group1_n=len(group1_values),
        group2_n=len(group2_values),
        pooled_std=pooled_std,
    )


def run_weat_computation(
    projection_df: pd.DataFrame, use_zscore: bool = False
) -> List[WEATResult]:
    """
    计算所有省份的WEAT效应量

    三个维度:
    1. work_family: d = (family_mean - work_mean) / pooled_std
       正值表示family更偏女性(传统性别规范)

    2. leadership: d = (non_leadership_mean - leadership_mean) / pooled_std
       正值表示leadership更偏男性(传统性别规范)

    3. stem: d = (non_stem_mean - stem_mean) / pooled_std
       正值表示stem更偏男性(传统性别规范)
    """
    print(f"\n{'='*70}")
    print(f"Step 4: 计算WEAT效应量 (Cohen's d)")
    print(f"{'='*70}\n")

    if use_zscore:
        print(f"  使用省内标准化后的z-score")
    else:
        print(f"  使用原始余弦相似度")

    provinces = projection_df["province"].unique()
    all_results = []

    dimensions = [
        ("work_family", "family", "work"),
        ("leadership", "non_leadership", "leadership"),
        ("stem", "non_stem", "stem"),
    ]

    for province in provinces:
        print(f"  处理省份: {province}")

        for dim_name, group1_cat, group2_cat in dimensions:
            result = compute_weat_for_dimension(
                projection_df,
                province,
                group1_cat,
                group2_cat,
                dim_name,
                use_zscore=use_zscore,
            )
            if result is not None:
                all_results.append(result)

    print(f"\n  WEAT效应量摘要:")
    results_df = pd.DataFrame([vars(r) for r in all_results])

    for dim_name, _, _ in dimensions:
        dim_results = results_df[results_df["dimension"] == dim_name]
        if len(dim_results) > 0:
            mean_d = dim_results["cohens_d"].mean()
            std_d = dim_results["cohens_d"].std()
            print(f"    - {dim_name}: 平均d = {mean_d:.3f} (SD = {std_d:.3f})")

    return all_results


# =============================================================================
# Step 5: 保存结果
# =============================================================================


def save_results(
    province_coverage_df: pd.DataFrame,
    word_coverage_df: pd.DataFrame,
    gender_axes: Dict[str, ProvinceGenderAxis],
    projection_df: pd.DataFrame,
    province_stats: List[ProvinceStats],
    weat_results: List[WEATResult],
    use_zscore: bool = False,
):
    """保存所有分析结果"""
    print(f"\n{'='*70}")
    print(f"Step 5: 保存结果")
    print(f"{'='*70}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    oov_file = os.path.join(OUTPUT_DIR, "oov_province_coverage.csv")
    province_coverage_df.to_csv(oov_file, index=False, encoding="utf-8-sig")
    print(f"  [保存] OOV省份覆盖率: {oov_file}")

    word_oov_file = os.path.join(OUTPUT_DIR, "oov_word_coverage.csv")
    word_coverage_df.to_csv(word_oov_file, index=False, encoding="utf-8-sig")
    print(f"  [保存] OOV词覆盖率: {word_oov_file}")

    gender_axis_data = []
    for province, axis in gender_axes.items():
        gender_axis_data.append(
            {
                "province": province,
                "n_male_words": len(axis.male_words_found),
                "n_female_words": len(axis.female_words_found),
                "male_words": ",".join(axis.male_words_found),
                "female_words": ",".join(axis.female_words_found),
            }
        )
    gender_axis_df = pd.DataFrame(gender_axis_data)
    gender_axis_file = os.path.join(OUTPUT_DIR, "gender_axes.csv")
    gender_axis_df.to_csv(gender_axis_file, index=False, encoding="utf-8-sig")
    print(f"  [保存] 性别轴信息: {gender_axis_file}")

    projection_file = os.path.join(OUTPUT_DIR, "word_projections.csv")
    projection_df.to_csv(projection_file, index=False, encoding="utf-8-sig")
    print(f"  [保存] 词投影结果: {projection_file}")

    stats_df = pd.DataFrame([vars(s) for s in province_stats])
    stats_file = os.path.join(OUTPUT_DIR, "province_projection_stats.csv")
    stats_df.to_csv(stats_file, index=False, encoding="utf-8-sig")
    print(f"  [保存] 省份投影统计: {stats_file}")

    weat_df = pd.DataFrame([vars(r) for r in weat_results])

    weat_wide = weat_df.pivot_table(
        index="province",
        columns="dimension",
        values=["cohens_d", "group1_n", "group2_n"],
        aggfunc="first",
    )
    weat_wide.columns = [f"{col[1]}_{col[0]}" for col in weat_wide.columns]
    weat_wide = weat_wide.reset_index()

    weat_file = os.path.join(OUTPUT_DIR, "weat_results.csv")
    weat_df.to_csv(weat_file, index=False, encoding="utf-8-sig")
    print(f"  [保存] WEAT详细结果: {weat_file}")

    weat_wide_file = os.path.join(OUTPUT_DIR, "gender_norm_index.csv")
    weat_wide.to_csv(weat_wide_file, index=False, encoding="utf-8-sig")
    print(f"  [保存] Gender Norm Index(主结果表): {weat_wide_file}")

    report_file = os.path.join(OUTPUT_DIR, "analysis_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"{'='*80}\n")
        f.write(f"Gender Norm Index 分析报告 (报纸数据, {DATA_YEAR}年)\n")
        f.write(f"{'='*80}\n\n")

        f.write(f"数据来源: 报纸语料库\n")
        f.write(f"分析省份数: {len(gender_axes)}\n")
        f.write(f"使用标准化: {'是' if use_zscore else '否'}\n\n")

        f.write(f"{'='*60}\n")
        f.write(f"WEAT效应量汇总 (Cohen's d)\n")
        f.write(f"{'='*60}\n\n")

        f.write(f"解释说明:\n")
        f.write(f"  - work_family: 正值 = family更偏女性(传统性别规范)\n")
        f.write(f"  - leadership: 正值 = leadership更偏男性(传统性别规范)\n")
        f.write(f"  - stem: 正值 = stem更偏男性(传统性别规范)\n")
        f.write(f"  - |d| ≈ 0.2 小效应, |d| ≈ 0.5 中效应, |d| ≈ 0.8 大效应\n\n")

        for dim in ["work_family", "leadership", "stem"]:
            dim_results = weat_df[weat_df["dimension"] == dim].sort_values(
                "cohens_d", ascending=False
            )
            if len(dim_results) > 0:
                f.write(f"\n{dim.upper()} 维度排名:\n")
                f.write(f"{'-'*50}\n")
                for i, (_, row) in enumerate(dim_results.iterrows(), 1):
                    f.write(
                        f"  {i:2d}. {row['province']:10s} | d = {row['cohens_d']:+.3f} "
                        f"| n1={row['group1_n']:2d}, n2={row['group2_n']:2d}\n"
                    )

    print(f"  [保存] 分析报告: {report_file}")
    print(f"\n  所有结果已保存到: {OUTPUT_DIR}/")


# =============================================================================
# 主函数
# =============================================================================


def main(standardize: bool = None, skip_oov: bool = False):
    """
    运行Gender Norm Index构建 (报纸数据版)

    Args:
        standardize: 是否进行省内标准化, None表示自动判断
        skip_oov: 是否跳过OOV检查(如果之前已运行过)
    """
    print(f"\n{'#'*70}")
    print(f"# Gender Norm Index 构建器 (报纸数据版)")
    print(f"# 数据年份: {DATA_YEAR}")
    print(f"{'#'*70}\n")

    wordlists = load_all_wordlists()

    if not skip_oov:
        province_coverage_df, word_coverage_df, _ = run_oov_check(wordlists)
        if province_coverage_df is None:
            return
    else:
        print(f"\n[跳过] OOV检查")
        province_coverage_df = pd.DataFrame()
        word_coverage_df = pd.DataFrame()

    gender_axes = run_build_gender_axes(wordlists)
    if not gender_axes:
        print(f"[错误] 未能构建任何性别轴")
        return

    projections = compute_all_projections(wordlists, gender_axes)
    if not projections:
        print(f"[错误] 未能计算任何投影")
        return

    province_stats, needs_standardization = run_comparability_check(projections)

    if standardize is None:
        use_zscore = needs_standardization
    else:
        use_zscore = standardize

    projection_df = pd.DataFrame([vars(p) for p in projections])
    if use_zscore:
        projection_df = standardize_projections(projections, province_stats)
        print(f"\n  [执行] 已进行省内z-score标准化")

    weat_results = run_weat_computation(projection_df, use_zscore=use_zscore)

    save_results(
        province_coverage_df=province_coverage_df,
        word_coverage_df=word_coverage_df,
        gender_axes=gender_axes,
        projection_df=projection_df,
        province_stats=province_stats,
        weat_results=weat_results,
        use_zscore=use_zscore,
    )

    print(f"\n{'#'*70}")
    print(f"# Gender Norm Index 构建完成!")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    fire.Fire(main)
