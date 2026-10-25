import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Set visualization style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "student_data.csv")
    output_dir = os.path.join(base_dir, "visual_outputs")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 60)
    print("[*] STUDENT PERFORMANCE CLUSTERING: MODEL TRAINING & ANALYSIS")
    print("=" * 60)

    # 1. Load Data
    print(f"\n[1] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    feature_cols = ["Attendance", "Internal_Marks", "Assignment_Marks", "Study_Hours"]
    X = df[feature_cols].copy()
    print(f"Loaded {len(df)} records with features: {feature_cols}")

    # 2. Feature Scaling
    print("\n[2] Applying StandardScaler normalization...")
    print("    Reason: Distance-based K-Means requires uniform feature scale")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Determining Optimal K (Elbow Method & Silhouette Score)
    print("\n[3] Evaluating Optimal K from 2 to 8 clusters...")
    k_range = range(1, 9)
    inertia_list = []
    silhouette_list = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertia_list.append(km.inertia_)
        if k >= 2:
            score = silhouette_score(X_scaled, km.labels_)
            silhouette_list.append((k, score))

    # Save Elbow Method Plot
    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), inertia_list, marker='o', linewidth=2.5, markersize=8, color='#1f77b4')
    plt.title("Elbow Method for Optimal K", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Number of Clusters (K)", fontsize=12)
    plt.ylabel("Within-Cluster Sum of Squares (Inertia)", fontsize=12)
    plt.axvline(x=3, color='#e74c3c', linestyle='--', label='Elbow Point at K=3')
    plt.legend(frameon=True)
    plt.tight_layout()
    elbow_path = os.path.join(output_dir, "elbow_method.png")
    plt.savefig(elbow_path, dpi=300)
    plt.close()
    print(f"    Saved Elbow plot to: {elbow_path}")

    # Save Silhouette Score Plot
    k_sil, scores_sil = zip(*silhouette_list)
    plt.figure(figsize=(8, 5))
    bars = plt.bar(k_sil, scores_sil, color='#2ca02c', alpha=0.85, width=0.55, edgecolor='#1b611b')
    plt.title("Silhouette Score for Different K Values", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Number of Clusters (K)", fontsize=12)
    plt.ylabel("Silhouette Score", fontsize=12)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom', fontsize=10)
    plt.ylim(0, max(scores_sil) + 0.1)
    plt.tight_layout()
    sil_path = os.path.join(output_dir, "silhouette_scores.png")
    plt.savefig(sil_path, dpi=300)
    plt.close()
    print(f"    Saved Silhouette plot to: {sil_path}")

    # 4. Train Final Model with Optimal K=3
    optimal_k = 3
    print(f"\n[4] Training KMeans with K={optimal_k}...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=15)
    cluster_labels = kmeans.fit_predict(X_scaled)
    df["Cluster_Raw"] = cluster_labels

    # 5. Dynamically Rank & Name Clusters based on Overall Performance
    # Calculate composite score for each cluster to assign intuitive labels
    cluster_stats = df.groupby("Cluster_Raw")[feature_cols].mean()
    composite_score = (
        cluster_stats["Attendance"] * 0.25 +
        cluster_stats["Internal_Marks"] * 0.35 +
        cluster_stats["Assignment_Marks"] * 0.25 +
        (cluster_stats["Study_Hours"] / 35.0 * 100) * 0.15
    )
    
    # Sort clusters from lowest to highest score
    sorted_clusters = composite_score.sort_values().index.tolist()
    
    label_map = {
        sorted_clusters[0]: {
            "name": "At-Risk / Needs Intervention",
            "tier": "Low Performance",
            "color": "#e74c3c",
            "recommendation": "Academic counseling, remedial sessions, mandatory attendance tracking, peer tutoring."
        },
        sorted_clusters[1]: {
            "name": "Moderate / Consistent Performers",
            "tier": "Average Performance",
            "color": "#f39c12",
            "recommendation": "Targeted problem-solving workshops, time-management mentoring, regular quiz practice."
        },
        sorted_clusters[2]: {
            "name": "High Achievers / Distinction",
            "tier": "Top Performance",
            "color": "#27ae60",
            "recommendation": "Advanced electives, competitive exam preparation, research mini-projects, mentoring peers."
        }
    }

    df["Cluster_Group"] = df["Cluster_Raw"].map(lambda c: label_map[c]["name"])
    df["Performance_Tier"] = df["Cluster_Raw"].map(lambda c: label_map[c]["tier"])

    print("\n[5] Cluster Summary Profiles (Mean Values):")
    for raw_id in sorted_clusters:
        info = label_map[raw_id]
        subset = df[df["Cluster_Raw"] == raw_id]
        print(f"\n  Group: {info['name']} (Count: {len(subset)})")
        print(f"    - Attendance:       {subset['Attendance'].mean():.1f}%")
        print(f"    - Internal Marks:   {subset['Internal_Marks'].mean():.1f}/100")
        print(f"    - Assignment Marks: {subset['Assignment_Marks'].mean():.1f}/100")
        print(f"    - Study Hours/Week: {subset['Study_Hours'].mean():.1f} hrs")
        print(f"    - Strategy:         {info['recommendation']}")

    # Save clustered data
    clustered_data_path = os.path.join(base_dir, "data", "clustered_students.csv")
    df.to_csv(clustered_data_path, index=False)
    print(f"\n[6] Saved labeled dataset to: {clustered_data_path}")

    # 6. Dimensionality Reduction & 2D PCA Visualization
    print("\n[7] Generating PCA 2D Cluster Visualizations...")
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)
    df["PC1"] = pca_coords[:, 0]
    df["PC2"] = pca_coords[:, 1]
    
    exp_var = pca.explained_variance_ratio_ * 100
    print(f"    Explained Variance: PC1 = {exp_var[0]:.1f}%, PC2 = {exp_var[1]:.1f}% (Total = {sum(exp_var):.1f}%)")

    plt.figure(figsize=(10, 7))
    for raw_id in sorted_clusters:
        info = label_map[raw_id]
        sub = df[df["Cluster_Raw"] == raw_id]
        plt.scatter(
            sub["PC1"], sub["PC2"],
            label=f"{info['name']} (n={len(sub)})",
            color=info["color"],
            alpha=0.75,
            edgecolors='k',
            linewidth=0.5,
            s=65
        )

    # Plot Cluster Centroids in PCA space
    pca_centroids = pca.transform(kmeans.cluster_centers_)
    for raw_id in sorted_clusters:
        plt.scatter(
            pca_centroids[raw_id, 0],
            pca_centroids[raw_id, 1],
            s=220,
            marker='X',
            color='black',
            edgecolors='white',
            linewidth=1.8,
            zorder=10
        )
    # Add fake legend entry for Centroids
    plt.scatter([], [], marker='X', color='black', s=120, label='Cluster Centroids')

    plt.title("Student Performance Clusters (PCA 2D Projection)", fontsize=15, fontweight='bold', pad=14)
    plt.xlabel(f"Principal Component 1 ({exp_var[0]:.1f}% Variance)", fontsize=12)
    plt.ylabel(f"Principal Component 2 ({exp_var[1]:.1f}% Variance)", fontsize=12)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10, loc='best')
    plt.tight_layout()
    pca_path = os.path.join(output_dir, "pca_clusters_2d.png")
    plt.savefig(pca_path, dpi=300)
    plt.close()
    print(f"    Saved PCA 2D plot to: {pca_path}")

    # 7. Feature Comparison Barplot
    plt.figure(figsize=(11, 6))
    summary_plot_data = df.groupby("Cluster_Group")[feature_cols].mean().reset_index()
    plot_melted = pd.melt(summary_plot_data, id_vars=["Cluster_Group"], var_name="Metric", value_name="Average_Value")
    
    palette = {"At-Risk / Needs Intervention": "#e74c3c", "Moderate / Consistent Performers": "#f39c12", "High Achievers / Distinction": "#27ae60"}
    sns.barplot(data=plot_melted, x="Metric", y="Average_Value", hue="Cluster_Group", palette=palette, edgecolor='black', alpha=0.9)
    plt.title("Comparison of Academic Metrics Across Performance Clusters", fontsize=14, fontweight='bold', pad=12)
    plt.ylabel("Average Value", fontsize=12)
    plt.xlabel("Student Feature", fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title="Cluster Group", frameon=True)
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, "cluster_profiles.png")
    plt.savefig(comparison_path, dpi=300)
    plt.close()
    print(f"    Saved Feature Comparison plot to: {comparison_path}")

    # 8. Feature Correlation Heatmap
    plt.figure(figsize=(7, 5.5))
    corr = df[feature_cols].corr()
    sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f", linewidths=1.2, square=True)
    plt.title("Correlation Matrix of Student Features", fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    corr_path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(corr_path, dpi=300)
    plt.close()
    print(f"    Saved Correlation Heatmap to: {corr_path}")

    # 9. Save Models & Mapping
    print("\n[8] Saving trained artifacts...")
    with open(os.path.join(models_dir, "kmeans_model.pkl"), "wb") as f:
        pickle.dump(kmeans, f)
    with open(os.path.join(models_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(models_dir, "pca.pkl"), "wb") as f:
        pickle.dump(pca, f)
        
    # Convert label_map keys to strings for JSON serialization
    json_label_map = {str(k): v for k, v in label_map.items()}
    with open(os.path.join(models_dir, "cluster_mapping.json"), "w") as f:
        json.dump(json_label_map, f, indent=4)
        
    print("    Saved:")
    print("    - models/kmeans_model.pkl")
    print("    - models/scaler.pkl")
    print("    - models/pca.pkl")
    print("    - models/cluster_mapping.json")
    print("\n" + "=" * 60)
    print("[SUCCESS] Model Training & Visualization Pipeline Completed Successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
