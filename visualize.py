"""
本地可视化脚本 - 读取 viz_data/ 中的parquet文件，生成PDF图表。

用法:
  python visualize.py all 2020
  python visualize.py retweet 2020
  python visualize.py interval 2020
  python visualize.py density_post 2020
  python visualize.py density_user 2020
"""

import os
import pandas as pd
import numpy as np
import fire
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.sans-serif"] = [
    "Arial Unicode MS",
    "SimHei",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False
from scipy.stats import gaussian_kde, sem, norm

VIZ_DIR = "viz_data"
FIG_DIR = "figures"

os.makedirs(FIG_DIR, exist_ok=True)

# Colors
MALE_COLOR = "#20AEE6"
FEMALE_COLOR = "#ff7333"

SOURCE_LABELS = {"news": "News", "entertain": "Entertainment"}
GENDER_ORDER = ["m", "f"]
GENDER_LABELS = {"m": "Male", "f": "Female"}
GENDER_COLORS = {"m": MALE_COLOR, "f": FEMALE_COLOR}


def _date_prefix():
    return datetime.now().strftime("%Y%m%d")


def _save_fig(fig, name):
    path = os.path.join(FIG_DIR, f"{_date_prefix()}_{name}.pdf")
    fig.savefig(path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  saved: {path}")


def _split_by_gender(df, col="retweet_count"):
    """Split data into male/female arrays."""
    m = df[df["gender"] == "m"][col].values
    f = df[df["gender"] == "f"][col].values
    return m, f


# ============================================================================
# Common plot helpers
# ============================================================================


def _kde_ecdf(ax, data_m, data_f, xlabel="", title="", log_x=False):
    """KDE + ECDF on twin y-axes.

    Left axis (ax): KDE density curves, prominent colored lines.
    Right axis (ax2): ECDF 0-1, light grey fill + thin colored lines.
    """
    if len(data_m) < 2 and len(data_f) < 2:
        ax.text(
            0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
        )
        ax.set_title(title)
        return

    ax2 = ax.twinx()

    # Build x grid
    all_vals = np.concatenate([d for d in [data_m, data_f] if len(d) >= 2])
    if log_x:
        all_vals = all_vals[all_vals > 0]
        if len(all_vals) < 2:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(title)
            return
        x_grid = np.logspace(
            np.log10(all_vals.min()), np.log10(all_vals.max()), 500
        )
    else:
        x_grid = np.linspace(all_vals.min(), all_vals.max(), 500)

    for gender in GENDER_ORDER:
        data = data_m if gender == "m" else data_f
        color = GENDER_COLORS[gender]
        label = GENDER_LABELS[gender]
        if len(data) < 2:
            continue

        # --- ECDF on right axis (behind KDE) ---
        sorted_d = np.sort(data)
        ecdf_y = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
        # Extend ECDF to full x-range (flat at 1.0 beyond data max)
        x_max = all_vals.max()
        if sorted_d[-1] < x_max:
            sorted_d = np.append(sorted_d, x_max)
            ecdf_y = np.append(ecdf_y, 1.0)
        # Subsample if too large (for PDF file size)
        if len(sorted_d) > 5000:
            idx = np.linspace(0, len(sorted_d) - 1, 5000, dtype=int)
            sorted_d_plot = sorted_d[idx]
            ecdf_y_plot = ecdf_y[idx]
        else:
            sorted_d_plot = sorted_d
            ecdf_y_plot = ecdf_y
        ax2.fill_between(
            sorted_d_plot, 0, ecdf_y_plot, alpha=0.08, color=color
        )
        ax2.plot(
            sorted_d_plot,
            ecdf_y_plot,
            color=color,
            alpha=0.4,
            linewidth=0.6,
            linestyle="-",
        )

        # --- KDE on left axis ---
        sample = data
        if len(data) > 100_000:
            rng = np.random.default_rng(42)
            sample = rng.choice(data, 100_000, replace=False)

        if log_x:
            kde = gaussian_kde(np.log(sample))
            kde_y = kde(np.log(x_grid))
        else:
            kde = gaussian_kde(sample)
            kde_y = kde(x_grid)
        ax.plot(x_grid, kde_y, color=color, linewidth=1.5, label=label)

    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density", fontsize=8)
    ax2.set_ylabel("ECDF", fontsize=8, color="#888888")
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis="y", colors="#888888", labelsize=7)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")


def _ci_plot(ax, data_m, data_f, ylabel="", title="", confidence=0.99, legend_loc="upper right"):
    """Mean + CI comparison (dot + whisker)."""
    z = norm.ppf(1 - (1 - confidence) / 2)

    positions = [0.3, 0.7]  # closer together, away from edges
    for i, gender in enumerate(GENDER_ORDER):
        data = data_m if gender == "m" else data_f
        color = GENDER_COLORS[gender]
        label = GENDER_LABELS[gender]
        if len(data) < 2:
            continue
        mean = np.mean(data)
        ci_half = z * sem(data)
        ax.errorbar(
            positions[i],
            mean,
            yerr=ci_half,
            fmt="o",
            color=color,
            markersize=8,
            capsize=6,
            capthick=1.5,
            linewidth=1.5,
            label=f"{label} (μ={mean:.4f})",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([GENDER_LABELS[g] for g in GENDER_ORDER])
    ax.set_xlim(0, 1)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, loc=legend_loc)


def _ratio_bar(ax, ratio_m, ratio_f, ylabel="Ratio", title=""):
    """Bar chart comparing two ratios."""
    bars = ax.bar(
        [0, 1],
        [ratio_m, ratio_f],
        color=[MALE_COLOR, FEMALE_COLOR],
        width=0.5,
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Male", "Female"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for bar, val in zip(bars, [ratio_m, ratio_f]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val*100:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )


# ============================================================================
# Part 1: Retweet count (3 versions)
# ============================================================================


def plot_retweet_count(year):
    """Generate 3 versions of retweet count figure (2 cols × 3 rows each)."""
    print("\n=== Retweet count figures ===")

    overview = pd.read_parquet(
        os.path.join(VIZ_DIR, f"retweet_overview_{year}.parquet")
    )
    overview = overview.set_index("gender")

    versions = {
        "v1": ("All retweeters (incl. 0)", None, None),
        "v2": ("Retweeters with count > 0", 0, None),
        "v3": ("Retweeters with count > 0, trimmed 95%", 0, 0.95),
    }

    for vname, (desc, exclude_zero, trim_q) in versions.items():
        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        fig.suptitle(f"Retweet Count Analysis — {desc}", fontsize=14, y=0.98)

        for col_idx, source_type in enumerate(["news", "entertain"]):
            src_label = SOURCE_LABELS[source_type]

            # Load data
            count_file = os.path.join(
                VIZ_DIR, f"retweet_counts_{source_type}_{year}.parquet"
            )
            if not os.path.exists(count_file):
                for row in range(3):
                    axes[row, col_idx].text(
                        0.5, 0.5, "No data", ha="center", va="center",
                        transform=axes[row, col_idx].transAxes,
                    )
                    axes[row, col_idx].set_title(f"{src_label}")
                continue

            df = pd.read_parquet(count_file)

            # --- Row 0: Ratio of retweeters ---
            retweet_col = f"users_with_{source_type}_retweet"
            ratio_m = (
                overview.loc["m", retweet_col]
                / overview.loc["m", "users_with_retweet"]
                if overview.loc["m", "users_with_retweet"] > 0
                else 0
            )
            ratio_f = (
                overview.loc["f", retweet_col]
                / overview.loc["f", "users_with_retweet"]
                if overview.loc["f", "users_with_retweet"] > 0
                else 0
            )
            _ratio_bar(
                axes[0, col_idx],
                ratio_m,
                ratio_f,
                ylabel="Ratio",
                title=f"{src_label} retweeters / all retweeters",
            )

            # --- Row 1: Distribution (version-specific filtering) ---
            plot_df = df.copy()
            if exclude_zero is not None:
                plot_df = plot_df[plot_df["retweet_count"] > 0]

            data_m, data_f = _split_by_gender(plot_df, "retweet_count")

            # Trim 95% tail per gender independently
            if trim_q is not None:
                if len(data_m) > 0:
                    data_m = data_m[data_m <= np.quantile(data_m, trim_q)]
                if len(data_f) > 0:
                    data_f = data_f[data_f <= np.quantile(data_f, trim_q)]

            use_log = exclude_zero is not None  # log x for non-zero versions
            _kde_ecdf(
                axes[1, col_idx],
                data_m,
                data_f,
                xlabel="Retweet count",
                title=f"{src_label} retweet count distribution",
                log_x=use_log,
            )

            # --- Row 2: Mean + 99% CI (same filtered data as distribution) ---
            _ci_plot(
                axes[2, col_idx],
                data_m,
                data_f,
                ylabel="Mean retweet count",
                title=f"{src_label} mean retweet count (99% CI)",
                legend_loc="upper left" if col_idx == 1 else "upper right",
            )

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        _save_fig(fig, f"retweet_count_{vname}")


# ============================================================================
# Part 1b: Retweet interval
# ============================================================================


def plot_retweet_interval(year):
    """Retweet interval figure (2 cols × 2 rows)."""
    print("\n=== Retweet interval figure ===")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Retweet Interval Analysis", fontsize=14, y=0.98)

    for col_idx, source_type in enumerate(["news", "entertain"]):
        src_label = SOURCE_LABELS[source_type]
        interval_file = os.path.join(
            VIZ_DIR, f"retweet_intervals_{source_type}_{year}.parquet"
        )

        if not os.path.exists(interval_file):
            for row in range(2):
                axes[row, col_idx].text(
                    0.5, 0.5, "No data", ha="center", va="center",
                    transform=axes[row, col_idx].transAxes,
                )
                axes[row, col_idx].set_title(f"{src_label}")
            continue

        df = pd.read_parquet(interval_file)
        # Convert to hours
        df["interval_hours"] = df["retweet_interval"] / 3600.0

        data_m, data_f = _split_by_gender(df, "interval_hours")

        # --- Row 0: Distribution ---
        _kde_ecdf(
            axes[0, col_idx],
            data_m,
            data_f,
            xlabel="Interval (hours)",
            title=f"{src_label} retweet interval distribution",
            log_x=True,
        )

        # --- Row 1: Mean + 99% CI ---
        _ci_plot(
            axes[1, col_idx],
            data_m,
            data_f,
            ylabel="Mean interval (hours)",
            title=f"{src_label} mean interval (99% CI)",
            legend_loc="upper left" if col_idx == 0 else "upper right",
        )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save_fig(fig, "retweet_interval")


# ============================================================================
# Part 2a: Post-level density
# ============================================================================


def plot_density_post(year):
    """Post-level density figure (2 cols × 3 rows)."""
    print("\n=== Post-level density figure ===")

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle("Post-Level Density Analysis", fontsize=14, y=0.98)

    summary = pd.read_parquet(
        os.path.join(VIZ_DIR, f"density_summary_{year}.parquet")
    )

    for col_idx, density_type in enumerate(["news", "entertain"]):
        src_label = SOURCE_LABELS[density_type]
        post_file = os.path.join(
            VIZ_DIR, f"density_post_{density_type}_{year}.parquet"
        )

        # --- Row 0: Ratio of density == 0 ---
        post_summary = summary[
            (summary["type"] == density_type) & (summary["level"] == "post")
        ]
        if len(post_summary) > 0:
            ratio_m = post_summary[post_summary["gender"] == "m"][
                "zero_ratio"
            ].values[0]
            ratio_f = post_summary[post_summary["gender"] == "f"][
                "zero_ratio"
            ].values[0]
        else:
            ratio_m, ratio_f = 0, 0

        _ratio_bar(
            axes[0, col_idx],
            ratio_m,
            ratio_f,
            ylabel="Ratio",
            title=f"{src_label} density = 0 / all posts",
        )

        # --- Row 1 & 2: Distribution + CI (density > 0) ---
        if not os.path.exists(post_file):
            for row in [1, 2]:
                axes[row, col_idx].text(
                    0.5, 0.5, "No data", ha="center", va="center",
                    transform=axes[row, col_idx].transAxes,
                )
                axes[row, col_idx].set_title(f"{src_label}")
            continue

        df = pd.read_parquet(post_file)  # already filtered to density > 0
        data_m, data_f = _split_by_gender(df, "density")

        _kde_ecdf(
            axes[1, col_idx],
            data_m,
            data_f,
            xlabel="Density",
            title=f"{src_label} post density distribution (> 0)",
            log_x=True,
        )

        _ci_plot(
            axes[2, col_idx],
            data_m,
            data_f,
            ylabel="Mean density",
            title=f"{src_label} mean post density (99% CI)",
            legend_loc="upper left" if col_idx == 1 else "upper right",
        )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save_fig(fig, "density_post")


# ============================================================================
# Part 2b: User-level density
# ============================================================================


def plot_density_user(year):
    """User-level density figure (2 cols × 3 rows)."""
    print("\n=== User-level density figure ===")

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle("User-Level Density Analysis", fontsize=14, y=0.98)

    summary = pd.read_parquet(
        os.path.join(VIZ_DIR, f"density_summary_{year}.parquet")
    )

    for col_idx, density_type in enumerate(["news", "entertain"]):
        src_label = SOURCE_LABELS[density_type]
        user_file = os.path.join(
            VIZ_DIR, f"density_user_{density_type}_{year}.parquet"
        )

        # --- Row 0: Ratio of avg_density == 0 ---
        user_summary = summary[
            (summary["type"] == density_type) & (summary["level"] == "user")
        ]
        if len(user_summary) > 0:
            ratio_m = user_summary[user_summary["gender"] == "m"][
                "zero_ratio"
            ].values[0]
            ratio_f = user_summary[user_summary["gender"] == "f"][
                "zero_ratio"
            ].values[0]
        else:
            ratio_m, ratio_f = 0, 0

        _ratio_bar(
            axes[0, col_idx],
            ratio_m,
            ratio_f,
            ylabel="Ratio",
            title=f"{src_label} avg density = 0 / all users",
        )

        # --- Row 1 & 2: Distribution + CI (avg_density > 0) ---
        if not os.path.exists(user_file):
            for row in [1, 2]:
                axes[row, col_idx].text(
                    0.5, 0.5, "No data", ha="center", va="center",
                    transform=axes[row, col_idx].transAxes,
                )
                axes[row, col_idx].set_title(f"{src_label}")
            continue

        df = pd.read_parquet(user_file)
        df_nz = df[df["avg_density"] > 0]
        data_m, data_f = _split_by_gender(df_nz, "avg_density")

        _kde_ecdf(
            axes[1, col_idx],
            data_m,
            data_f,
            xlabel="Avg density",
            title=f"{src_label} user avg density distribution (> 0)",
            log_x=True,
        )

        _ci_plot(
            axes[2, col_idx],
            data_m,
            data_f,
            ylabel="Mean avg density",
            title=f"{src_label} mean user density (99% CI)",
            legend_loc="upper left" if col_idx == 1 else "upper right",
        )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save_fig(fig, "density_user")


# ============================================================================
# Entry points
# ============================================================================


def retweet(year):
    plot_retweet_count(year)
    plot_retweet_interval(year)


def density(year):
    plot_density_post(year)
    plot_density_user(year)


def all(year):
    plot_retweet_count(year)
    plot_retweet_interval(year)
    plot_density_post(year)
    plot_density_user(year)
    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    fire.Fire(
        {
            "all": all,
            "retweet": retweet,
            "interval": plot_retweet_interval,
            "density": density,
            "density_post": plot_density_post,
            "density_user": plot_density_user,
        }
    )
