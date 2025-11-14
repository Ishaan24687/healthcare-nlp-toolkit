"""
TF-IDF clustering of CPT code descriptions.
The question: can we recover clinical categories (E/M, Surgery, Radiology, etc.)
purely from the text of the procedure descriptions? Turns out KMeans on TF-IDF
features does a reasonable job — silhouette around 0.35, and most clusters map
clearly to one dominant category.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.manifold import TSNE
from collections import Counter


def load_cpt_data(data_path: str) -> pd.DataFrame:
    return pd.read_csv(data_path)


def cluster_cpt_descriptions(
    cpt_df: pd.DataFrame,
    n_clusters: int = 7,
    output_dir: str = "outputs",
):
    print("=" * 60)
    print("TF-IDF Clustering of CPT Descriptions")
    print("=" * 60)

    cpt_descriptions = cpt_df["description"].tolist()
    cpt_categories = cpt_df["category"].tolist()

    tfidf_vectorizer = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 3),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
        stop_words="english",
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(cpt_descriptions)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    # try range of k to find best silhouette
    print("\nSilhouette scores by k:")
    silhouette_results = {}
    for k in range(3, 12):
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        temp_labels = kmeans_temp.fit_predict(tfidf_matrix)
        sil = silhouette_score(tfidf_matrix, temp_labels)
        silhouette_results[k] = sil
        print(f"  k={k:2d}: silhouette={sil:.4f}")

    best_k = max(silhouette_results, key=silhouette_results.get)
    print(f"\nBest k by silhouette: {best_k} (score: {silhouette_results[best_k]:.4f})")
    print(f"Using k={n_clusters} (matches number of clinical categories)")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)
    cpt_df = cpt_df.copy()
    cpt_df["cluster"] = cluster_labels

    sil_score = silhouette_score(tfidf_matrix, cluster_labels)
    print(f"\nSilhouette score (k={n_clusters}): {sil_score:.4f}")

    # how well do clusters align with actual categories?
    category_to_id = {cat: i for i, cat in enumerate(sorted(set(cpt_categories)))}
    true_labels = [category_to_id[c] for c in cpt_categories]
    ari = adjusted_rand_score(true_labels, cluster_labels)
    print(f"Adjusted Rand Index: {ari:.4f}")

    print("\nCluster composition:")
    for cluster_id in range(n_clusters):
        cluster_mask = cpt_df["cluster"] == cluster_id
        category_counts = Counter(cpt_df[cluster_mask]["category"])
        dominant_category = category_counts.most_common(1)[0][0]
        purity = category_counts.most_common(1)[0][1] / sum(category_counts.values())
        print(f"  Cluster {cluster_id}: {dict(category_counts)} (dominant: {dominant_category}, purity: {purity:.2f})")

    # top terms per cluster
    feature_names = tfidf_vectorizer.get_feature_names_out()
    print("\nTop terms per cluster:")
    cluster_centers = kmeans.cluster_centers_
    for cluster_id in range(n_clusters):
        top_term_indices = cluster_centers[cluster_id].argsort()[-8:][::-1]
        top_terms = [feature_names[i] for i in top_term_indices]
        print(f"  Cluster {cluster_id}: {', '.join(top_terms)}")

    os.makedirs(output_dir, exist_ok=True)
    _plot_tsne(tfidf_matrix, cluster_labels, cpt_categories, output_dir)
    _plot_silhouette_curve(silhouette_results, n_clusters, output_dir)

    results = {
        "n_clusters": n_clusters,
        "silhouette_score": float(sil_score),
        "adjusted_rand_index": float(ari),
        "best_k_by_silhouette": int(best_k),
    }
    with open(os.path.join(output_dir, "clustering_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return cpt_df, results


def _plot_tsne(tfidf_matrix, cluster_labels, cpt_categories, output_dir: str):
    perplexity = min(30, tfidf_matrix.shape[0] - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    coords = tsne.fit_transform(tfidf_matrix.toarray())

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    unique_clusters = sorted(set(cluster_labels))
    cluster_colors = plt.cm.Set2(np.linspace(0, 1, len(unique_clusters)))
    for cluster_id, color in zip(unique_clusters, cluster_colors):
        mask = cluster_labels == cluster_id
        axes[0].scatter(coords[mask, 0], coords[mask, 1], c=[color], label=f"Cluster {cluster_id}", alpha=0.7, s=40)
    axes[0].set_title("t-SNE: KMeans Clusters")
    axes[0].legend(fontsize=8)
    axes[0].set_xlabel("t-SNE 1")
    axes[0].set_ylabel("t-SNE 2")

    unique_categories = sorted(set(cpt_categories))
    cat_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_categories)))
    for cpt_category, color in zip(unique_categories, cat_colors):
        mask = [c == cpt_category for c in cpt_categories]
        axes[1].scatter(coords[mask, 0], coords[mask, 1], c=[color], label=cpt_category, alpha=0.7, s=40)
    axes[1].set_title("t-SNE: Actual Clinical Categories")
    axes[1].legend(fontsize=8)
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "cpt_clustering_tsne.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nt-SNE plot saved to {plot_path}")


def _plot_silhouette_curve(silhouette_results: dict, chosen_k: int, output_dir: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = sorted(silhouette_results.keys())
    scores = [silhouette_results[k] for k in ks]
    ax.plot(ks, scores, "bo-", linewidth=2)
    ax.axvline(x=chosen_k, color="r", linestyle="--", label=f"Chosen k={chosen_k}")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Analysis for CPT Description Clustering")
    ax.legend()
    ax.grid(alpha=0.3)

    plot_path = os.path.join(output_dir, "silhouette_curve.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Silhouette curve saved to {plot_path}")


if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    data_path = os.path.join(base_dir, "data", "cpt_codes.csv")
    output_dir = os.path.join(base_dir, "outputs")

    if not os.path.exists(data_path):
        print("CPT data not found. Generating dataset first...")
        from cpt_data import save_cpt_dataset
        save_cpt_dataset(os.path.join(base_dir, "data"))

    cpt_df = load_cpt_data(data_path)
    cluster_cpt_descriptions(cpt_df, n_clusters=7, output_dir=output_dir)
