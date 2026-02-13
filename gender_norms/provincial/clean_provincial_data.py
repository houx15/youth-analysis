from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

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


def _clean_text(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace(" ", "").replace("\u3000", "").strip()


def normalize_province(name: object) -> str | None:
    text = _clean_text(name)
    if not text or text.lower() == "nan":
        return None
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


def load_gdp(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="gbk", skiprows=3)
    df["province"] = df["地区"].map(normalize_province)
    df = df[df["province"].notna()].copy()
    df["gdp_2024"] = pd.to_numeric(df["2024年"], errors="coerce")
    return df[["province", "gdp_2024"]]


def load_income(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="gbk", skiprows=3)
    df["province"] = df["地区"].map(normalize_province)
    df = df[df["province"].notna()].copy()
    df["avg_income_2024"] = pd.to_numeric(df["2024年"], errors="coerce")
    return df[["province", "avg_income_2024"]]


def load_cfps(path: Path) -> pd.DataFrame:
    df = pd.read_stata(path)
    if "provcd20" not in df.columns or "gender_ideation" not in df.columns:
        raise ValueError("cfps_2020_gender_ideation.dta missing required columns.")
    df = df[["provcd20", "gender_ideation"]].copy()
    df["province"] = df["provcd20"].map(normalize_province)
    df = df[df["province"].notna()].copy()

    df["gender_ideation"] = pd.to_numeric(df["gender_ideation"], errors="coerce")
    grouped = (
        df.groupby("province", as_index=False)
        .agg(obs_count=("gender_ideation", "count"), cfps_ideation_2020=("gender_ideation", "mean"))
    )
    grouped.loc[grouped["obs_count"] <= 100, "cfps_ideation_2020"] = pd.NA
    return grouped[["province", "cfps_ideation_2020"]]


def load_cfps_gender(path: Path) -> pd.DataFrame:
    df = pd.read_stata(path)
    for col in ("gender", "provcd20", "is_leader", "daily_housework_hours", "daily_rear_hours"):
        if col not in df.columns:
            raise ValueError(f"cfps_2020_gender_data.dta missing required column: {col}")

    df["province"] = df["provcd20"].map(normalize_province)
    df = df[df["province"].notna()].copy()

    df["is_leader"] = pd.to_numeric(df["is_leader"], errors="coerce")
    df["daily_housework_hours"] = pd.to_numeric(df["daily_housework_hours"], errors="coerce")
    df["daily_rear_hours"] = pd.to_numeric(df["daily_rear_hours"], errors="coerce")

    # is_leader: P(leader | male) - P(leader | female) per province
    leader = df[df["is_leader"].notna()].copy()
    leader_by = leader.groupby(["province", "gender"], as_index=False)["is_leader"].mean()
    leader_m = leader_by[leader_by["gender"] == "男"].rename(columns={"is_leader": "leader_m"})
    leader_f = leader_by[leader_by["gender"] == "女"].rename(columns={"is_leader": "leader_f"})
    leader_merged = leader_m[["province", "leader_m"]].merge(leader_f[["province", "leader_f"]], on="province")
    leader_merged["leader_gap_mf"] = leader_merged["leader_m"] - leader_merged["leader_f"]

    # daily_housework_hours: female - male per province
    hw = df[df["daily_housework_hours"].notna()].copy()
    hw_by = hw.groupby(["province", "gender"], as_index=False)["daily_housework_hours"].mean()
    hw_m = hw_by[hw_by["gender"] == "男"].rename(columns={"daily_housework_hours": "hw_m"})
    hw_f = hw_by[hw_by["gender"] == "女"].rename(columns={"daily_housework_hours": "hw_f"})
    hw_merged = hw_m[["province", "hw_m"]].merge(hw_f[["province", "hw_f"]], on="province")
    hw_merged["housework_gap_fm"] = hw_merged["hw_f"] - hw_merged["hw_m"]

    # daily_rear_hours: female - male per province
    rear = df[df["daily_rear_hours"].notna()].copy()
    rear_by = rear.groupby(["province", "gender"], as_index=False)["daily_rear_hours"].mean()
    rear_m = rear_by[rear_by["gender"] == "男"].rename(columns={"daily_rear_hours": "rear_m"})
    rear_f = rear_by[rear_by["gender"] == "女"].rename(columns={"daily_rear_hours": "rear_f"})
    rear_merged = rear_m[["province", "rear_m"]].merge(rear_f[["province", "rear_f"]], on="province")
    rear_merged["rear_gap_fm"] = rear_merged["rear_f"] - rear_merged["rear_m"]

    result = leader_merged[["province", "leader_gap_mf"]].merge(
        hw_merged[["province", "housework_gap_fm"]], on="province", how="outer"
    ).merge(
        rear_merged[["province", "rear_gap_fm"]], on="province", how="outer"
    )
    return result


def _build_multilevel_columns(df: pd.DataFrame, header_row: int) -> list[str]:
    top = df.iloc[header_row].ffill()
    sub = df.iloc[header_row + 1].fillna("")
    columns: list[str] = []
    for idx in range(df.shape[1]):
        if idx == 0:
            columns.append("地区")
            continue
        top_text = _clean_text(top.iloc[idx])
        sub_text = _clean_text(sub.iloc[idx])
        if sub_text:
            columns.append(f"{top_text}_{sub_text}")
        else:
            columns.append(top_text)
    return columns


def _find_header_row(df: pd.DataFrame, target: str) -> int:
    first_col = df.iloc[:, 0].map(_clean_text)
    matches = df.index[first_col == target].tolist()
    if not matches:
        raise ValueError(f"Header row with target '{target}' not found.")
    return matches[0]


def load_education(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, header=None)
    header_row = _find_header_row(df, "地区")
    columns = _build_multilevel_columns(df, header_row)
    data = df.iloc[header_row + 2 :].copy()
    data.columns = columns

    data["地区"] = data["地区"].map(_clean_text)
    data = data[~data["地区"].isin({"", "全国", "全国", "全  国", "全    国", "男", "女", "性别", "性    别"})]
    data["province"] = data["地区"].map(normalize_province)
    data = data[data["province"].notna()].copy()

    edu_years = {
        "未上过学": 0,
        "学前教育": 0,
        "小学": 6,
        "初中": 9,
        "高中": 12,
        "大学专科": 15,
        "大学本科": 16,
        "硕士研究生": 19,
        "博士研究生": 23,
    }

    total_col = "25岁及以上人口_合计"
    male_col = "25岁及以上人口_男"
    female_col = "25岁及以上人口_女"

    def weighted_average(prefix: str, denominator: str) -> pd.Series:
        total = 0
        for level, years in edu_years.items():
            col = f"{level}_{prefix}"
            total += pd.to_numeric(data[col], errors="coerce") * years
        denom = pd.to_numeric(data[denominator], errors="coerce")
        return total / denom

    data["eduy_gt25_2020"] = weighted_average("小计", total_col)
    data["eduy_m_gt25_2020"] = weighted_average("男", male_col)
    data["eduy_f_gt25_2020"] = weighted_average("女", female_col)

    return data[["province", "eduy_gt25_2020", "eduy_m_gt25_2020", "eduy_f_gt25_2020"]]


def load_population(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, header=None)
    header_row = _find_header_row(df, "地区")
    columns = _build_multilevel_columns(df, header_row)
    data = df.iloc[header_row + 2 :].copy()
    data.columns = columns

    data["地区"] = data["地区"].map(_clean_text)
    data = data[~data["地区"].isin({"", "全国", "全 国", "全    国", "男", "女", "性别", "性    别"})]
    data["province"] = data["地区"].map(normalize_province)
    data = data[data["province"].notna()].copy()

    def sum_age_groups(sex: str, groups: list[str]) -> pd.Series:
        cols = [f"{group}_{sex}" for group in groups]
        return data[cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)

    male_groups = [
        "15-19岁",
        "20-24岁",
        "25-29岁",
        "30-34岁",
        "35-39岁",
        "40-44岁",
        "45-49岁",
        "50-54岁",
        "55-59岁",
        "60-64岁",
    ]
    female_groups = [
        "15-19岁",
        "20-24岁",
        "25-29岁",
        "30-34岁",
        "35-39岁",
        "40-44岁",
        "45-49岁",
        "50-54岁",
        "55-59岁",
    ]

    data["pop_m_16_60"] = sum_age_groups("男", male_groups)
    data["pop_f_16_55"] = sum_age_groups("女", female_groups)
    data["pop_16_60_55_total"] = data["pop_m_16_60"] + data["pop_f_16_55"]
    return data[["province", "pop_m_16_60", "pop_f_16_55", "pop_16_60_55_total"]]


def load_employment(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, header=None)
    header_row = _find_header_row(df, "地区")
    columns = _build_multilevel_columns(df, header_row)
    data = df.iloc[header_row + 2 :].copy()
    data.columns = columns
    data["地区"] = data["地区"].map(_clean_text)

    idx_male = data.index[data["地区"] == "男"].tolist()
    idx_female = data.index[data["地区"] == "女"].tolist()
    if not idx_male or not idx_female:
        raise ValueError("Male/Female section markers not found in employment file.")
    idx_male = idx_male[0]
    idx_female = idx_female[0]

    total = data.loc[: idx_male - 1].copy()
    male = data.loc[idx_male + 1 : idx_female - 1].copy()
    female = data.loc[idx_female + 1 :].copy()

    def clean_section(section: pd.DataFrame) -> pd.DataFrame:
        section = section[~section["地区"].isin({"", "全国", "全 国", "全    国"})].copy()
        section["province"] = section["地区"].map(normalize_province)
        return section[section["province"].notna()].copy()

    total = clean_section(total)
    male = clean_section(male)
    female = clean_section(female)

    male_groups = [
        "16-19岁",
        "20-24岁",
        "25-29岁",
        "30-34岁",
        "35-39岁",
        "40-44岁",
        "45-49岁",
        "50-54岁",
        "55-59岁",
        "60-64岁",
    ]
    female_groups = [
        "16-19岁",
        "20-24岁",
        "25-29岁",
        "30-34岁",
        "35-39岁",
        "40-44岁",
        "45-49岁",
        "50-54岁",
        "55-59岁",
    ]

    def sum_groups(section: pd.DataFrame, groups: list[str]) -> pd.Series:
        return section[groups].apply(pd.to_numeric, errors="coerce").sum(axis=1)

    male["emp_m_16_60"] = sum_groups(male, male_groups)
    female["emp_f_16_55"] = sum_groups(female, female_groups)

    merged = male[["province", "emp_m_16_60"]].merge(
        female[["province", "emp_f_16_55"]],
        on="province",
        how="outer",
    )
    merged["emp_16_60_55_total"] = merged["emp_m_16_60"] + merged["emp_f_16_55"]
    return merged


def main() -> None:
    base = Path(__file__).resolve().parent
    gdp = load_gdp(base / "gdp.csv")
    income = load_income(base / "income.csv")
    edu = load_education(base / "gender_education_25.xls")
    pop = load_population(base / "gender_population.xls")
    emp = load_employment(base / "gender_employment.xls")
    cfps = load_cfps(base / "cfps_2020_gender_ideation.dta")
    cfps_gender = load_cfps_gender(base / "cfps_2020_gender_data.dta")

    merged = gdp.merge(income, on="province", how="outer")
    merged = merged.merge(edu, on="province", how="outer")
    merged = merged.merge(pop, on="province", how="outer")
    merged = merged.merge(emp, on="province", how="outer")
    merged = merged.merge(cfps, on="province", how="outer")
    merged = merged.merge(cfps_gender, on="province", how="outer")

    merged["emp_m_2020"] = merged["emp_m_16_60"] / merged["pop_m_16_60"]
    merged["emp_f_2020"] = merged["emp_f_16_55"] / merged["pop_f_16_55"]
    merged["emp_2020"] = merged["emp_16_60_55_total"] / merged["pop_16_60_55_total"]

    output = merged[
        [
            "province",
            "gdp_2024",
            "avg_income_2024",
            "eduy_gt25_2020",
            "eduy_m_gt25_2020",
            "eduy_f_gt25_2020",
            "emp_2020",
            "emp_m_2020",
            "emp_f_2020",
            "cfps_ideation_2020",
            "leader_gap_mf",
            "housework_gap_fm",
            "rear_gap_fm",
        ]
    ].copy()

    output = output.sort_values("province")
    output_path = base / "provincial_cleaned.csv"
    output.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
