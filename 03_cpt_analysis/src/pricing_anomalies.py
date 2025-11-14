"""
Anomaly detection on CPT pricing using both Isolation Forest and z-score methods.
Flags CPT codes with unusual reimbursement patterns — either codes that are priced
way off from their category peers, or codes with abnormally large regional spread.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats


def load_cpt_data(data_path: str) -> pd.DataFrame:
    return pd.read_csv(data_path)


def prepare_pricing_features(cpt_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Build a feature matrix for anomaly detection. Includes the raw regional rates
    plus derived features like regional spread and deviation from category mean.
    """
    region_cols = [
        "reimbursement_northeast", "reimbursement_south",
        "reimbursement_midwest", "reimbursement_west",
    ]

    features = cpt_df[region_cols].copy()
    features["regional_spread"] = features[region_cols].max(axis=1) - features[region_cols].min(axis=1)
    features["regional_cv"] = features[region_cols].std(axis=1) / features[region_cols].mean(axis=1)
    features["national_avg"] = cpt_df["national_avg_reimbursement"]

    # deviation from category mean — a $500 procedure in a category that averages
    # $100 is more interesting than a $500 procedure in a $450 category
    category_means = cpt_df.groupby("category")["national_avg_reimbursement"].transform("mean")
    category_stds = cpt_df.groupby("category")["national_avg_reimbursement"].transform("std")
    features["category_z_score"] = (cpt_df["national_avg_reimbursement"] - category_means) / category_stds.replace(0, 1)

    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(features)

    return features, feature_matrix


def detect_anomalies_isolation_forest(
    cpt_df: pd.DataFrame,
    feature_matrix: np.ndarray,
    contamination: float = 0.1,
) -> pd.DataFrame:
    print("=" * 60)
    print("Isolation Forest Anomaly Detection")
    print("=" * 60)

    iso_forest = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        max_features=0.8,
    )
    anomaly_labels = iso_forest.fit_predict(feature_matrix)
    anomaly_scores = iso_forest.decision_function(feature_matrix)

    cpt_df = cpt_df.copy()
    cpt_df["anomaly_label"] = anomaly_labels  # -1 = anomaly, 1 = normal
    cpt_df["anomaly_score"] = anomaly_scores

    anomalous_cpt_codes = cpt_df[cpt_df["anomaly_label"] == -1].sort_values("anomaly_score")

    print(f"\nDetected {len(anomalous_cpt_codes)} anomalous CPT codes:")
    print(f"{'CPT':>7} {'Category':>12} {'Natl Avg':>10} {'NE':>8} {'South':>8} {'MW':>8} {'West':>8} {'Score':>8}")
    print("-" * 80)

    for _, row in anomalous_cpt_codes.iterrows():
        print(
            f"{row['cpt_code']:>7} {row['category']:>12} "
            f"${row['national_avg_reimbursement']:>8.2f} "
            f"${row['reimbursement_northeast']:>6.2f} "
            f"${row['reimbursement_south']:>6.2f} "
            f"${row['reimbursement_midwest']:>6.2f} "
            f"${row['reimbursement_west']:>6.2f} "
            f"{row['anomaly_score']:>8.4f}"
        )

    return cpt_df


def detect_anomalies_zscore(cpt_df: pd.DataFrame, threshold: float = 2.0) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print(f"Z-Score Anomaly Detection (threshold={threshold})")
    print("=" * 60)

    cpt_df = cpt_df.copy()
    region_cols = [
        "reimbursement_northeast", "reimbursement_south",
        "reimbursement_midwest", "reimbursement_west",
    ]

    # z-score within each category — comparing to category peers is more meaningful
    # than comparing a pathology test to a total knee replacement
    category_anomalies = []
    for cpt_category in sorted(cpt_df["category"].unique()):
        category_mask = cpt_df["category"] == cpt_category
        category_data = cpt_df[category_mask]

        for col in region_cols:
            z_scores = np.abs(stats.zscore(category_data[col].values))
            anomaly_indices = category_data.index[z_scores > threshold]
            for idx in anomaly_indices:
                category_anomalies.append({
                    "cpt_code": cpt_df.loc[idx, "cpt_code"],
                    "category": cpt_category,
                    "region": col.replace("reimbursement_", "").title(),
                    "rate": float(cpt_df.loc[idx, col]),
                    "z_score": float(z_scores[list(category_data.index).index(idx)]),
                    "category_mean": float(category_data[col].mean()),
                    "category_std": float(category_data[col].std()),
                })

    print(f"\nFound {len(category_anomalies)} category-level pricing anomalies:")
    for anomaly in sorted(category_anomalies, key=lambda x: -x["z_score"])[:15]:
        direction = "above" if anomaly["rate"] > anomaly["category_mean"] else "below"
        print(
            f"  {anomaly['cpt_code']} ({anomaly['category']}) in {anomaly['region']}: "
            f"${anomaly['rate']:.2f} vs category avg ${anomaly['category_mean']:.2f} "
            f"(z={anomaly['z_score']:.2f}, {direction})"
        )

    return pd.DataFrame(category_anomalies) if category_anomalies else pd.DataFrame()


def plot_anomalies(cpt_df: pd.DataFrame, output_dir: str):
    if "anomaly_label" not in cpt_df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    normal_mask = cpt_df["anomaly_label"] == 1
    anomaly_mask = cpt_df["anomaly_label"] == -1

    axes[0].scatter(
        cpt_df[normal_mask]["national_avg_reimbursement"],
        cpt_df[normal_mask]["reimbursement_south"],
        c="#2196F3", alpha=0.6, s=40, label="Normal",
    )
    axes[0].scatter(
        cpt_df[anomaly_mask]["national_avg_reimbursement"],
        cpt_df[anomaly_mask]["reimbursement_south"],
        c="#f44336", alpha=0.8, s=80, marker="x", linewidths=2, label="Anomaly",
    )
    axes[0].plot(
        [0, cpt_df["national_avg_reimbursement"].max()],
        [0, cpt_df["national_avg_reimbursement"].max()],
        "k--", alpha=0.3, label="y=x (parity)",
    )
    axes[0].set_xlabel("National Average ($)")
    axes[0].set_ylabel("South Reimbursement ($)")
    axes[0].set_title("South vs National Average — Anomalies Highlighted")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    anomaly_counts = cpt_df[anomaly_mask]["category"].value_counts()
    total_counts = cpt_df["category"].value_counts()
    anomaly_pct = (anomaly_counts / total_counts * 100).fillna(0).sort_values(ascending=True)

    anomaly_pct.plot(kind="barh", ax=axes[1], color="#FF9800", alpha=0.85)
    axes[1].set_xlabel("% Anomalous CPT Codes")
    axes[1].set_title("Anomaly Rate by Category")
    axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "pricing_anomalies.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nAnomaly plots saved to {plot_path}")


def run_anomaly_detection(data_path: str, output_dir: str = "outputs"):
    cpt_df = load_cpt_data(data_path)
    print(f"Loaded {len(cpt_df)} CPT codes\n")

    os.makedirs(output_dir, exist_ok=True)

    features, feature_matrix = prepare_pricing_features(cpt_df)
    cpt_df = detect_anomalies_isolation_forest(cpt_df, feature_matrix, contamination=0.1)
    zscore_anomalies = detect_anomalies_zscore(cpt_df, threshold=2.0)

    plot_anomalies(cpt_df, output_dir)

    # combine and summarize
    n_iso_anomalies = (cpt_df["anomaly_label"] == -1).sum()
    n_zscore_anomalies = len(zscore_anomalies)

    # overlap analysis
    iso_anomaly_codes = set(cpt_df[cpt_df["anomaly_label"] == -1]["cpt_code"])
    zscore_anomaly_codes = set(zscore_anomalies["cpt_code"]) if len(zscore_anomalies) > 0 else set()
    both_methods = iso_anomaly_codes & zscore_anomaly_codes

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Isolation Forest anomalies: {n_iso_anomalies}")
    print(f"Z-score anomalies: {n_zscore_anomalies}")
    print(f"Flagged by both methods: {len(both_methods)}")
    if both_methods:
        print(f"  Codes: {', '.join(sorted(both_methods))}")

    summary = {
        "isolation_forest_anomalies": n_iso_anomalies,
        "zscore_anomalies": n_zscore_anomalies,
        "flagged_by_both": list(sorted(both_methods)),
    }
    with open(os.path.join(output_dir, "anomaly_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    data_path = os.path.join(base_dir, "data", "cpt_codes.csv")
    output_dir = os.path.join(base_dir, "outputs")

    if not os.path.exists(data_path):
        print("CPT data not found. Generating dataset first...")
        from cpt_data import save_cpt_dataset
        save_cpt_dataset(os.path.join(base_dir, "data"))

    run_anomaly_detection(data_path, output_dir)
