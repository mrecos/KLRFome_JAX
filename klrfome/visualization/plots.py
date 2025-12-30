"""Visualization utilities for KLRfome."""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from jaxtyping import Array, Float


def plot_similarity_matrix(
    K: Float[Array, "n n"],
    labels: Optional[Float[Array, "n"]] = None,
    n_clusters: int = 4,
    figsize: tuple = (10, 8),
    cmap: str = 'viridis'
):
    """
    Plot similarity matrix with hierarchical clustering.
    
    Similar to R's K_corrplot function.
    
    Parameters:
        K: Similarity matrix
        labels: Optional labels for coloring
        n_clusters: Number of clusters for visualization
        figsize: Figure size
        cmap: Colormap
    """
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    from scipy.spatial.distance import squareform
    
    K_np = np.array(K)
    
    # Convert similarity to distance for clustering
    # Higher similarity = lower distance
    max_sim = np.max(K_np)
    distances = max_sim - K_np
    
    # Hierarchical clustering
    condensed_distances = squareform(distances, checks=False)
    linkage_matrix = linkage(condensed_distances, method='ward')
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    
    # Reorder matrix by clusters
    order = np.argsort(cluster_labels)
    K_ordered = K_np[order][:, order]
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(K_ordered, cmap=cmap, aspect='auto')
    ax.set_title('Similarity Matrix (Clustered)')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Sample Index')
    
    plt.colorbar(im, ax=ax, label='Similarity')
    plt.tight_layout()
    
    return fig, ax


def plot_predictions(
    predictions: Float[Array, "height width"],
    sites: Optional = None,
    figsize: tuple = (10, 8),
    cmap: str = 'viridis',
    title: str = 'Prediction Map'
):
    """
    Plot prediction raster.
    
    Parameters:
        predictions: 2D prediction array
        sites: Optional GeoDataFrame with site locations to overlay
        figsize: Figure size
        cmap: Colormap
        title: Plot title
    """
    pred_np = np.array(predictions)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(pred_np, cmap=cmap, origin='upper')
    ax.set_title(title)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    
    # Overlay sites if provided
    if sites is not None:
        try:
            import geopandas as gpd
            # Convert to pixel coordinates if needed
            # This is a simplified version
            for idx, row in sites.iterrows():
                # Would need transform to convert to pixel coords
                pass
        except ImportError:
            pass
    
    plt.colorbar(im, ax=ax, label='Probability')
    plt.tight_layout()
    
    return fig, ax


def plot_roc_curve(
    pred: Float[Array, "n"],
    obs: Float[Array, "n"],
    figsize: tuple = (8, 8)
):
    """
    Plot ROC curve.
    
    Parameters:
        pred: Predicted probabilities
        obs: Observed labels (1/0)
        figsize: Figure size
    
    Returns:
        Figure and axes
    """
    from sklearn.metrics import roc_curve, auc
    
    pred_np = np.array(pred)
    obs_np = np.array(obs)
    
    fpr, tpr, thresholds = roc_curve(obs_np, pred_np)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig, ax

