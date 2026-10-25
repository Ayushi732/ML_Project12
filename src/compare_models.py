import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "student_data.csv")
    output_dir = os.path.join(base_dir, "visual_outputs")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 65)
    print("[*] CLUSTERING ALGORITHM BENCHMARK: K-MEANS vs AGGLOMERATIVE vs GMM")
    print("=" * 65)

    df = pd.read_csv(data_path)
    feature_cols = ["Attendance", "Internal_Marks", "Assignment_Marks", "Study_Hours"]
    X = df[feature_cols].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k = 3
    models = {
        "K-Means": KMeans(n_clusters=k, random_state=42, n_init=10),
        "Agglomerative (Hierarchical)": AgglomerativeClustering(n_clusters=k, linkage='ward'),
        "Gaussian Mixture (GMM)": GaussianMixture(n_components=k, random_state=42)
    }

    results = []

    for name, model in models.items():
        if name == "Gaussian Mixture (GMM)":
            labels = model.fit_predict(X_scaled)
        else:
            labels = model.fit_predict(X_scaled)

        sil = silhouette_score(X_scaled, labels)
        db = davies_bouldin_score(X_scaled, labels)
        ch = calinski_harabasz_score(X_scaled, labels)

        results.append({
            "Algorithm": name,
            "Silhouette Score (Higher Better)": round(sil, 4),
            "Davies-Bouldin Index (Lower Better)": round(db, 4),
            "Calinski-Harabasz Score (Higher Better)": round(ch, 2)
        })

    results_df = pd.DataFrame(results)
    print("\nBenchmark Results Summary Table:")
    print(results_df.to_string(index=False))

    # Plot Comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    # Silhouette Score
    sns.barplot(data=results_df, x="Algorithm", y="Silhouette Score (Higher Better)", ax=axes[0], palette=palette)
    axes[0].set_title("Silhouette Score\n(Higher is Better)", fontsize=11, fontweight='bold')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=20, ha='right')
    axes[0].set_ylabel("Score")

    # Davies-Bouldin Index
    sns.barplot(data=results_df, x="Algorithm", y="Davies-Bouldin Index (Lower Better)", ax=axes[1], palette=palette)
    axes[1].set_title("Davies-Bouldin Index\n(Lower is Better)", fontsize=11, fontweight='bold')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=20, ha='right')
    axes[1].set_ylabel("Index")

    # Calinski-Harabasz Score
    sns.barplot(data=results_df, x="Algorithm", y="Calinski-Harabasz Score (Higher Better)", ax=axes[2], palette=palette)
    axes[2].set_title("Calinski-Harabasz Score\n(Higher is Better)", fontsize=11, fontweight='bold')
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=20, ha='right')
    axes[2].set_ylabel("Score")

    plt.tight_layout()
    comp_plot_path = os.path.join(output_dir, "algorithm_comparison.png")
    plt.savefig(comp_plot_path, dpi=300)
    plt.close()
    print(f"\n[+] Saved algorithm comparison plot to: {comp_plot_path}")
    print("=" * 65)

if __name__ == "__main__":
    main()
