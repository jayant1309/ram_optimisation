"""
Step 3: Exploratory Data Analysis and Visualization

This module performs comprehensive EDA including distribution analysis,
correlation heatmaps, PCA dimensionality reduction, t-SNE visualization,
and K-Means clustering to understand the structure and patterns in the
engineered materials features.
"""

import os
import sys

# Detect environment - must be at top of every file
try:
    import google.colab
    IN_COLAB = True
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_DIR = '/content/drive/MyDrive/ram_optimisation'
except ImportError:
    IN_COLAB = False
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, 'data')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.append(BASE_DIR)
from config import FEATURE_COLS, TARGET_COL, RANDOM_STATE

def plot_target_distribution(df):
    """
    Plot histogram and KDE of target variable (e_total).

    Args:
        df (pd.DataFrame): DataFrame with e_total column

    Returns:
        matplotlib.figure.Figure: Figure object
    """
    print("Now at Step 3a] Plotting target distribution...")

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Plot histogram with KDE overlay
    sns.histplot(data=df, x=TARGET_COL, kde=True, ax=ax, color='steelblue')
    ax.set_title(f'Distribution of {TARGET_COL}', fontsize=14, fontweight='bold')
    ax.set_xlabel(TARGET_COL, fontsize=12)
    ax.set_ylabel('Count', fontsize=12)

    # Add summary statistics
    mean_val = df[TARGET_COL].mean()
    median_val = df[TARGET_COL].median()
    std_val = df[TARGET_COL].std()

    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
    ax.axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')
    ax.legend()

    plt.tight_layout()
    output_path = os.path.join(PLOTS_DIR, 'eda_target_distribution.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    if not IN_COLAB:
        plt.show()

    plt.close('all')
    return fig


def plot_correlation_heatmap(df):
    """
    Plot correlation heatmap of all features and target.

    Args:
        df (pd.DataFrame): DataFrame with features and target

    Returns:
        matplotlib.figure.Figure: Figure object
    """
    print("Now at Step 3b] Plotting correlation heatmap...")

    plt.style.use('seaborn-v0_8-darkgrid')

    # Select numeric columns only
    numeric_cols = FEATURE_COLS + [TARGET_COL]
    corr_matrix = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)

    # Plot heatmap
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                square=True, cbar_kws={'shrink': 0.8}, ax=ax)
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(PLOTS_DIR, 'eda_correlation_heatmap.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    if not IN_COLAB:
        plt.show()

    plt.close('all')
    return fig


def plot_boxplot_by_crystal_system(df):
    """
    Plot box plot of target values grouped by crystal system.

    Args:
        df (pd.DataFrame): DataFrame with crystal_system_encoded

    Returns:
        matplotlib.figure.Figure: Figure object
    """
    print("Now at Step 3c] Plotting box plot by crystal system...")

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Get crystal systems from the encoded values
    crystal_systems = sorted(df['crystal_system_encoded'].unique())

    box_data = [df[df['crystal_system_encoded'] == cs][TARGET_COL].values for cs in crystal_systems]

    ax.boxplot(box_data, labels=[f'CS_{cs}' for cs in crystal_systems])
    ax.set_title(f'{TARGET_COL} Distribution by Crystal System', fontsize=14, fontweight='bold')
    ax.set_xlabel('Crystal System (Encoded)', fontsize=12)
    ax.set_ylabel(TARGET_COL, fontsize=12)

    plt.tight_layout()
    output_path = os.path.join(PLOTS_DIR, 'eda_boxplot_crystal.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    if not IN_COLAB:
        plt.show()

    plt.close('all')
    return fig


def plot_pca_scatter(df):
    """
    Perform PCA dimensionality reduction and plot scatter plot.

    Args:
        df (pd.DataFrame): DataFrame with features

    Returns:
        tuple: (matplotlib.figure.Figure, explained_variance)
    """
    print("Now at Step 3d] Performing PCA and plotting scatter...")

    plt.style.use('seaborn-v0_8-darkgrid')

    # Fit PCA on features only
    scaler = StandardScaler()
    feature_data = scaler.fit_transform(df[FEATURE_COLS])

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_result = pca.fit_transform(feature_data)

    explained_variance = pca.explained_variance_ratio_
    print(f"PCA explained variance ratio: PC1={explained_variance[0]:.3f}, PC2={explained_variance[1]:.3f}")

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1],
                        c=df[TARGET_COL], cmap='viridis',
                        alpha=0.6, s=30)

    ax.set_xlabel(f'Principal Component 1 ({explained_variance[0]:.1%})', fontsize=12)
    ax.set_ylabel(f'Principal Component 2 ({explained_variance[1]:.1%})', fontsize=12)
    ax.set_title('PCA: Materials Colored by Dielectric Response', fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(TARGET_COL, rotation=270, labelpad=20, fontsize=12)

    plt.tight_layout()
    output_path = os.path.join(PLOTS_DIR, 'pca_scatter.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    if not IN_COLAB:
        plt.show()

    plt.close('all')
    return fig, explained_variance


def plot_tsne_scatter(df):
    """
    Perform t-SNE dimensionality reduction and plot scatter plot.

    Args:
        df (pd.DataFrame): DataFrame with features

    Returns:
        matplotlib.figure.Figure: Figure object
    """
    print("Now at Step 3e] Performing t-SNE and plotting scatter...")

    plt.style.use('seaborn-v0_8-darkgrid')

    # Prepare feature data
    scaler = StandardScaler()
    feature_data = scaler.fit_transform(df[FEATURE_COLS])

    # Perform t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=RANDOM_STATE,
                learning_rate='auto', init='pca')
    tsne_result = tsne.fit_transform(feature_data)

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    scatter = ax.scatter(tsne_result[:, 0], tsne_result[:, 1],
                        c=df[TARGET_COL], cmap='viridis',
                        alpha=0.6, s=30)

    ax.set_xlabel('t-SNE Component 1', fontsize=12)
    ax.set_ylabel('t-SNE Component 2', fontsize=12)
    ax.set_title('t-SNE: Materials Colored by Dielectric Response', fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(TARGET_COL, rotation=270, labelpad=20, fontsize=12)

    plt.tight_layout()
    output_path = os.path.join(PLOTS_DIR, 'tsne_scatter.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    if not IN_COLAB:
        plt.show()

    plt.close('all')
    return fig


def plot_kmeans_clustering(df):
    """
    Perform K-Means clustering with elbow curve and plot results.

    Args:
        df (pd.DataFrame): DataFrame with features

    Returns:
        tuple: (elbow_fig, cluster_fig, df_with_clusters)
    """
    print("Now at Step 3f] Performing K-Means clustering...")

    plt.style.use('seaborn-v0_8-darkgrid')
    # Prepare feature data
    scaler = StandardScaler()
    feature_data = scaler.fit_transform(df[FEATURE_COLS])

    # Elbow curve
    print("  Computing elbow curve (k=2 to 10)...")
    inertia = []
    k_range = range(2, 11)

    for k in tqdm(k_range, desc="K-Means for elbow curve"):
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init='auto')
        kmeans.fit(feature_data)
        inertia.append(kmeans.inertia_)

    # Plot elbow curve
    fig_elbow, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.plot(k_range, inertia, 'o-', color='steelblue', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
    ax.set_title('K-Means Elbow Curve', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Find optimal k (simple heuristic: find "elbow" where improvement slows)
    # Use the point where the decrease in inertia is less than 20% of the maximum decrease
    diffs = np.diff(inertia)
    if len(diffs) > 0:
        max_decrease_idx = np.argmin(diffs)
        optimal_k = k_range[max_decrease_idx + 1] if max_decrease_idx < len(k_range) - 1 else 3
    else:
        optimal_k = 3

    print(f"  Selected optimal k: {optimal_k}")

    # Save elbow plot
    elbow_path = os.path.join(PLOTS_DIR, 'kmeans_elbow.png')
    plt.savefig(elbow_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {elbow_path}")

    if not IN_COLAB:
        plt.show()

    plt.close('all')

    # Fit final K-Means with optimal k
    print(f"  Fitting K-Means with k={optimal_k}...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init='auto')
    clusters = kmeans.fit_predict(feature_data)

    # Add cluster labels to dataframe
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = clusters

    # Print cluster statistics
    print("\n  Cluster statistics:")
    for cluster_id in range(optimal_k):
        cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
        mean_target = cluster_data[TARGET_COL].mean()
        count = len(cluster_data)
        print(f"    Cluster {cluster_id}: {count} materials, mean {TARGET_COL} = {mean_target:.3f}")

    # Plot clusters on PCA axes
    print("  Plotting clusters on PCA axes...")
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_result = pca.fit_transform(feature_data)

    fig_cluster, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Plot each cluster with different color
    colors = plt.cm.Set3(np.linspace(0, 1, optimal_k))
    for cluster_id in range(optimal_k):
        cluster_mask = clusters == cluster_id
        ax.scatter(pca_result[cluster_mask, 0], pca_result[cluster_mask, 1],
                  label=f'Cluster {cluster_id}', alpha=0.6, s=30)

    ax.set_xlabel(f'Principal Component 1', fontsize=12)
    ax.set_ylabel(f'Principal Component 2', fontsize=12)
    ax.set_title(f'K-Means Clusters (k={optimal_k}) on PCA Axes', fontsize=14, fontweight='bold')
    ax.legend()

    # Save cluster plot
    cluster_path = os.path.join(PLOTS_DIR, 'kmeans_clusters.png')
    plt.savefig(cluster_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {cluster_path}")

    if not IN_COLAB:
        plt.show()

    plt.close('all')

    # Save clustered dataframe
    cluster_df_path = os.path.join(DATA_DIR, 'clustered_features.csv')
    df_with_clusters.to_csv(cluster_df_path, index=False)
    print(f"Saved clustered features to: {cluster_df_path}")

    return fig_elbow, fig_cluster, df_with_clusters


def perform_eda_and_visualization(df):
    """
    Perform complete EDA and generate all visualizations.

    Args:
        df (pd.DataFrame): DataFrame with features and target

    Returns:
        dict: Dictionary of generated plots and data
    """
    print("=" * 60)
    print("STEP 3: EXPLORATORY DATA ANALYSIS AND VISUALIZATION")
    print("=" * 60)

    # Create all visualizations
    outputs = {}

    # 3a: Target distribution
    outputs['target_distribution'] = plot_target_distribution(df)

    # 3b: Correlation heatmap
    outputs['correlation_heatmap'] = plot_correlation_heatmap(df)

    # 3c: Box plot by crystal system
    outputs['boxplot_crystal'] = plot_boxplot_by_crystal_system(df)

    # 3d: PCA scatter plot
    outputs['pca_scatter'], pca_variance = plot_pca_scatter(df)
    outputs['pca_variance'] = pca_variance

    # 3e: t-SNE scatter plot
    outputs['tsne_scatter'] = plot_tsne_scatter(df)

    # 3f: K-Means clustering
    outputs['kmeans_elbow'], outputs['kmeans_clusters'], df_clustered = plot_kmeans_clustering(df)

    print(f"Step 3 complete All visualizations saved to {PLOTS_DIR}")

    return outputs, df_clustered


if __name__ == '__main__':
    import sys

    print("Loading engineered features...")
    input_path = os.path.join(DATA_DIR, 'features.csv')
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} materials with {len(df.columns)} features")

    try:
        outputs, df_clustered = perform_eda_and_visualization(df)
        print(f"\nStep 3 complete EDA completed successfully!")
    except Exception as e:
        print(f"[ERROR] EDA failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
