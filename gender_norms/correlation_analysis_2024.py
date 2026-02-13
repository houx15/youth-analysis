from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from scipy.stats import pearsonr
except Exception:  # pragma: no cover - optional dependency
    pearsonr = None

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
except Exception:
    pass

INPUT_DIR = Path("gender_norms/results")
OUTPUT_DIR = Path("gender_norms/results/2024/correlation")

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

PROVINCE_REVERSE = {v: k for k, v in PROVINCE_NAME_MAPPING.items()}


def normalize_province(name: object) -> str | None:
    text = "" if name is None else str(name)
    text = text.replace(" ", "").replace("\u3000", "").strip()
    if not text or text.lower() == "nan":
        return None
    if text.startswith("中国"):
        text = text.replace("中国", "", 1)
    if text in PROVINCE_NAME_MAPPING:
        return text
    if text in PROVINCE_REVERSE:
        return PROVINCE_REVERSE[text]
    for suffix in (
        "省",
        "市",
        "自治区",
        "特别行政区",
        "壮族自治区",
        "回族自治区",
        "维吾尔自治区",
    ):
        if text.endswith(suffix):
            stripped = text[: -len(suffix)]
            if stripped in PROVINCE_NAME_MAPPING:
                return stripped
            if stripped in PROVINCE_REVERSE:
                return PROVINCE_REVERSE[stripped]
    return None


def generate_output_path(year: int, filename: str) -> str:
    """生成输出文件路径"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    date_prefix = datetime.now().strftime("%Y%m%d")
    return os.path.join(OUTPUT_DIR, f"{date_prefix}_{filename}.pdf")


def compute_corr(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    data = pd.concat([x, y], axis=1).dropna()
    if data.empty or len(data) < 3:
        return np.nan, np.nan
    if pearsonr is None:
        return float(data.corr().iloc[0, 1]), np.nan
    r, p = pearsonr(data.iloc[:, 0], data.iloc[:, 1])
    return float(r), float(p)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    norms = pd.read_csv("gender_norms/results/2024/gender_norm_index.csv")
    prov = pd.read_csv("gender_norms/provincial/provincial_cleaned.csv")

    norms["province"] = norms["province"].map(normalize_province)
    prov["province"] = prov["province"].map(normalize_province)

    merged = norms.merge(prov, on="province", how="inner")
    merged["log_gdp_2024"] = merged["gdp_2024"].apply(
        lambda v: np.log(v) if pd.notna(v) and v > 0 else np.nan
    )
    merged["log_avg_income_2024"] = merged["avg_income_2024"].apply(
        lambda v: np.log(v) if pd.notna(v) and v > 0 else np.nan
    )
    merged["emp_diff_m_f"] = merged["emp_m_2020"] - merged["emp_f_2020"]
    merged["eduy_diff_m_f"] = merged["eduy_m_gt25_2020"] - merged["eduy_f_gt25_2020"]

    var_map = [
        ("log_gdp_2024", "log(GDP)"),
        ("eduy_gt25_2020", "Average Years of Education Attainment"),
        ("emp_2020", "Employment Rate"),
        ("log_avg_income_2024", "log(Average Income)"),
        ("eduy_diff_m_f", "Difference in Education Years (Male - Female)"),
        ("emp_diff_m_f", "Difference in Employment Rate (Male - Female)"),
        ("cfps_ideation_2020", "Gender Ideation in Survey (CFPS)"),
        ("leader_gap_mf", "Gap in P(Leader): Male − Female (CFPS)"),
        ("housework_gap_fm", "Extra Daily Housework Hours: Female − Male (CFPS)"),
        ("rear_gap_fm", "Extra Daily Childcare Hours: Female − Male (CFPS)"),
    ]

    cohens_cols = [
        ("leadership_cohens_d", "Leadership Gender Norm from Weibo (WEAT, Cohen's d)"),
        ("stem_cohens_d", "STEM Gender Norm from Weibo (WEAT, Cohen's d)"),
        (
            "work_family_cohens_d",
            "Work-Family Gender Norm from Weibo (WEAT, Cohen's d)",
        ),
    ]

    # Correlation results
    rows = []
    for coh_col, _ in cohens_cols:
        for var_col, _ in var_map:
            r, p = compute_corr(merged[coh_col], merged[var_col])
            n = merged[[coh_col, var_col]].dropna().shape[0]
            rows.append(
                {
                    "cohens_d": coh_col,
                    "variable": var_col,
                    "n": n,
                    "pearson_r": r,
                    "p_value": p,
                }
            )
    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUTPUT_DIR / "correlation_results.csv", index=False)

    # sns.set_theme(style="whitegrid")

    for coh_col, coh_title in cohens_cols:
        nrows = (len(var_map) + 2) // 3
        fig, axes = plt.subplots(nrows, 3, figsize=(16, 4 * nrows))
        axes = axes.flatten()
        for ax, (var_col, var_label) in zip(axes, var_map):
            plot_df = merged[[coh_col, var_col]].dropna()
            if plot_df.empty:
                ax.set_axis_off()
                continue
            sns.regplot(
                data=plot_df,
                x=var_col,
                y=coh_col,
                ax=ax,
                scatter_kws={"s": 30, "alpha": 0.8},
                line_kws={"color": "red"},
            )
            r, p = compute_corr(plot_df[var_col], plot_df[coh_col])
            ax.set_title(f"{var_label}\n$r$={r:.3f}, p={p:.3f}")
            ax.set_xlabel(var_label)
            ax.set_ylabel(coh_title)

        for ax in axes[len(var_map) :]:
            ax.set_axis_off()

        fig.tight_layout()
        filename = f"{coh_col}_correlation"
        output_path = generate_output_path(2024, filename)
        fig.savefig(output_path, format="pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
