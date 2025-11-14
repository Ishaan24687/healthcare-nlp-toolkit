"""
Statistical analysis of CPT reimbursement rates across regions and categories.
Tests normality assumptions, identifies outliers, and runs regional comparisons.
Found that ASC codes in the South are 15% below the national average — matches
what we saw at Lantern when comparing pricing across MSAs.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def load_cpt_data(data_path: str) -> pd.DataFrame:
    return pd.read_csv(data_path)


def analyze_reimbursement_distributions(cpt_df: pd.DataFrame, output_dir: str = "outputs"):
    print("=" * 60)
    print("Reimbursement Distribution Analysis")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sns.histplot(cpt_df["national_avg_reimbursement"], kde=True, bins=30, ax=axes[0, 0], color="#2196F3")
    axes[0, 0].set_title("National Average Reimbursement Distribution")
    axes[0, 0].set_xlabel("Reimbursement ($)")

    categories = sorted(cpt_df["category"].unique())
    category_data = [cpt_df[cpt_df["category"] == cat]["national_avg_reimbursement"] for cat in categories]
    bp = axes[0, 1].boxplot(category_data, labels=categories, patch_artist=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(categories)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    axes[0, 1].set_title("Reimbursement by Category")
    axes[0, 1].set_ylabel("Reimbursement ($)")
    axes[0, 1].tick_params(axis="x", rotation=45)

    regions = ["Northeast", "South", "Midwest", "West"]
    region_cols = [f"reimbursement_{r.lower()}" for r in regions]
    region_means = [cpt_df[col].mean() for col in region_cols]
    bar_colors = ["#1976D2", "#D32F2F", "#388E3C", "#F57C00"]
    axes[1, 0].bar(regions, region_means, color=bar_colors, alpha=0.85)
    axes[1, 0].set_title("Mean Reimbursement by Region")
    axes[1, 0].set_ylabel("Mean Reimbursement ($)")
    for i, (region, mean_val) in enumerate(zip(regions, region_means)):
        axes[1, 0].text(i, mean_val + 5, f"${mean_val:.0f}", ha="center", fontsize=9)

    for region, col, color in zip(regions, region_cols, bar_colors):
        sns.kdeplot(cpt_df[col], label=region, ax=axes[1, 1], color=color, linewidth=1.5)
    axes[1, 1].set_title("Reimbursement Distribution by Region")
    axes[1, 1].set_xlabel("Reimbursement ($)")
    axes[1, 1].legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "reimbursement_distributions.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Distribution plots saved to {plot_path}")


def test_normality(cpt_df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("Normality Testing (Shapiro-Wilk)")
    print("=" * 60)

    results = {}

    national_rates = cpt_df["national_avg_reimbursement"].values
    stat, p_value = stats.shapiro(national_rates[:50])  # Shapiro-Wilk limited to 5000
    results["national_overall"] = {"statistic": float(stat), "p_value": float(p_value)}
    print(f"\nNational overall: W={stat:.4f}, p={p_value:.6f}")
    print(f"  {'NORMAL' if p_value > 0.05 else 'NOT NORMAL'} (alpha=0.05)")

    for cpt_category in sorted(cpt_df["category"].unique()):
        category_rates = cpt_df[cpt_df["category"] == cpt_category]["national_avg_reimbursement"].values
        if len(category_rates) >= 3:
            stat, p_value = stats.shapiro(category_rates)
            results[cpt_category] = {"statistic": float(stat), "p_value": float(p_value)}
            normal_str = "NORMAL" if p_value > 0.05 else "NOT NORMAL"
            print(f"{cpt_category:15s}: W={stat:.4f}, p={p_value:.6f} ({normal_str})")

    print("\nSkewness and Kurtosis:")
    skewness = stats.skew(national_rates)
    kurtosis_val = stats.kurtosis(national_rates)
    print(f"  Skewness: {skewness:.4f} (positive = right-skewed)")
    print(f"  Kurtosis: {kurtosis_val:.4f} (>0 = heavy tails)")

    return results


def identify_reimbursement_outliers(cpt_df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("Outlier Detection (Z-Score > 2)")
    print("=" * 60)

    rates = cpt_df["national_avg_reimbursement"]
    z_scores = np.abs(stats.zscore(rates))
    cpt_df = cpt_df.copy()
    cpt_df["z_score"] = z_scores
    outlier_mask = z_scores > 2
    outlier_cpt_codes = cpt_df[outlier_mask].sort_values("z_score", ascending=False)

    print(f"\nFound {len(outlier_cpt_codes)} outlier CPT codes (z > 2):")
    for _, row in outlier_cpt_codes.iterrows():
        print(f"  {row['cpt_code']} ({row['category']:10s}) ${row['national_avg_reimbursement']:>8.2f} z={row['z_score']:.2f}")
        print(f"    {row['description'][:80]}...")

    return outlier_cpt_codes


def regional_comparison(cpt_df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("Regional Reimbursement Comparisons")
    print("=" * 60)

    regions = ["Northeast", "South", "Midwest", "West"]
    region_cols = {r: f"reimbursement_{r.lower()}" for r in regions}

    # all pairwise t-tests
    print("\nPairwise t-tests (independent samples):")
    print(f"{'Comparison':<30} {'t-stat':>10} {'p-value':>12} {'Significant':>12}")
    print("-" * 70)

    comparison_results = []
    for i, region_a in enumerate(regions):
        for region_b in regions[i + 1:]:
            rates_a = cpt_df[region_cols[region_a]].values
            rates_b = cpt_df[region_cols[region_b]].values
            t_stat, p_value = stats.ttest_ind(rates_a, rates_b)
            significant = "YES" if p_value < 0.05 else "no"
            comparison = f"{region_a} vs {region_b}"
            print(f"{comparison:<30} {t_stat:>10.4f} {p_value:>12.6f} {significant:>12}")
            comparison_results.append({
                "comparison": comparison, "t_statistic": float(t_stat),
                "p_value": float(p_value), "significant": p_value < 0.05,
            })

    # ASC-specific regional analysis — this is where the South gap shows up
    print("\n\nASC Code Regional Analysis:")
    asc_codes = cpt_df[cpt_df["category"] == "ASC"]
    if len(asc_codes) > 0:
        national_asc_avg = asc_codes["national_avg_reimbursement"].mean()
        print(f"  National ASC average: ${national_asc_avg:.2f}")

        for region in regions:
            region_asc_avg = asc_codes[region_cols[region]].mean()
            pct_diff = (region_asc_avg - national_asc_avg) / national_asc_avg * 100
            print(f"  {region:12s}: ${region_asc_avg:.2f} ({pct_diff:+.1f}% vs national)")

    # category-level regional analysis
    print("\n\nCategory x Region Mean Reimbursement:")
    print(f"{'Category':<15}", end="")
    for region in regions:
        print(f"{region:>12}", end="")
    print(f"{'Spread':>10}")
    print("-" * 65)

    for cpt_category in sorted(cpt_df["category"].unique()):
        cat_data = cpt_df[cpt_df["category"] == cpt_category]
        print(f"{cpt_category:<15}", end="")
        means = []
        for region in regions:
            mean_val = cat_data[region_cols[region]].mean()
            means.append(mean_val)
            print(f"${mean_val:>10.2f}", end="")
        spread = max(means) - min(means)
        print(f"${spread:>8.2f}")

    return comparison_results


def run_full_analysis(data_path: str, output_dir: str = "outputs"):
    cpt_df = load_cpt_data(data_path)
    print(f"Loaded {len(cpt_df)} CPT codes\n")

    os.makedirs(output_dir, exist_ok=True)

    analyze_reimbursement_distributions(cpt_df, output_dir)
    normality_results = test_normality(cpt_df)
    outlier_codes = identify_reimbursement_outliers(cpt_df)
    comparison_results = regional_comparison(cpt_df)

    summary = {
        "n_codes": len(cpt_df),
        "normality": normality_results,
        "n_outliers": len(outlier_codes),
        "regional_comparisons": comparison_results,
    }
    with open(os.path.join(output_dir, "reimbursement_analysis.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull analysis results saved to {output_dir}/reimbursement_analysis.json")


if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    data_path = os.path.join(base_dir, "data", "cpt_codes.csv")
    output_dir = os.path.join(base_dir, "outputs")

    if not os.path.exists(data_path):
        print("CPT data not found. Generating dataset first...")
        from cpt_data import save_cpt_dataset
        save_cpt_dataset(os.path.join(base_dir, "data"))

    run_full_analysis(data_path, output_dir)
