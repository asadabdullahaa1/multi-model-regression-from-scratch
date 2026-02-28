

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns

from utils.metrics import mae, r2_score, rmse


# Set publication-quality style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_clusters(X, labels, title="Cluster Visualization", save_path=None):
    """
    Visualize K-Means clusters in 2D using PCA
    """
    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # Get unique labels
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot each cluster with different color
    colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                  c=[colors[i]], label=f'Cluster {label}',
                  alpha=0.6, s=30, edgecolors='black', linewidth=0.5)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_model_weights(weights, cluster_sizes=None, title="WMMLR Model Weights",
                       save_path=None):
    """
    Visualize WMMLR weights with enhanced information
    """
    weights = np.asarray(weights, dtype=float)
    k = len(weights)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(k)
    bars = ax.bar(x, weights, color='steelblue', alpha=0.7,
                  edgecolor='black', linewidth=1.5)

    # Color bars based on weight magnitude
    max_weight = np.max(weights) if k > 0 else 0.0
    if max_weight > 0:
        color_scale = weights / max_weight
    else:
        color_scale = np.zeros_like(weights)
    colors = plt.cm.RdYlGn(color_scale)
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    # Add value labels on bars
    for i, (bar, weight) in enumerate(zip(bars, weights)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{weight:.4f}', ha='center', va='bottom',
               fontsize=11, fontweight='bold')

        # Add cluster size if provided
        if cluster_sizes is not None:
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                   f'n={cluster_sizes[i]}', ha='center', va='center',
                   fontsize=9, color='white', fontweight='bold')

    ax.set_xlabel('Cluster Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Weight', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Cluster {i}' for i in range(k)])
    ax.grid(True, alpha=0.3, axis='y')

    # Add horizontal line for equal weighting
    equal_weight = 1.0 / k
    ax.axhline(y=equal_weight, color='red', linestyle='--',
               linewidth=2, label=f'Equal Weight ({equal_weight:.3f})')
    ax.legend(fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()



def compare_model_errors(rmse_single, rmse_mmlr, rmse_wmmlr, metric_name="RMSE"):
    """
    Compare all three models errors visually
    """
    plt.figure(figsize=(8, 6))

    models = ["Single LR", "MMLR", "WMMLR"]
    values = [rmse_single, rmse_mmlr, rmse_wmmlr]
    colors = ["coral", "gold", "steelblue"]

    bars = plt.bar(models, values, color=colors, alpha=0.7, edgecolor='black')

    plt.ylabel(metric_name, fontsize=12)
    plt.title(f"{metric_name} Comparison Across Models", fontsize=14)

    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f"{value:.3f}", ha='center', va='bottom', fontsize=11)

    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

def plot_predictions_vs_actual(y_true, y_pred, model_name, dataset_name,
                               save_path=None):
    """
    Scatter plot of predictions vs actual values with statistics
    """
    #from utils.metrics import rmse, mae, r2_score

    fig, ax = plt.subplots(figsize=(8, 8))

    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.5, s=30, edgecolors='black', linewidth=0.5)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val],
            'r--', lw=3, label='Perfect Prediction', alpha=0.8)

    # Calculate metrics
    rmse_val = rmse(y_true, y_pred)
    mae_val = mae(y_true, y_pred)
    r2_val = r2_score(y_true, y_pred)

    # Add text box with metrics
    textstr = f'RMSE: {rmse_val:.4f}\nMAE:  {mae_val:.4f}\nR2:   {r2_val:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props, fontfamily='monospace')

    ax.set_xlabel('Actual Values', fontsize=13, fontweight='bold')
    ax.set_ylabel('Predicted Values', fontsize=13, fontweight='bold')
    ax.set_title(f'{model_name} - {dataset_name}\nPredictions vs Actual',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Make it square
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_residuals(y_true, y_pred, model_name, save_path=None):
    """
    Plot residuals to check for patterns
    """
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residual plot
    axes[0].scatter(y_pred, residuals, alpha=0.5, s=30)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted Values', fontsize=12)
    axes[0].set_ylabel('Residuals', fontsize=12)
    axes[0].set_title(f'{model_name} - Residual Plot', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Histogram of residuals
    axes[1].hist(residuals, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residuals', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title(f'{model_name} - Residual Distribution',
                     fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_cluster_performance(cluster_analysis, save_path=None):
    """
    """
    n_clusters = len(cluster_analysis)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    clusters = [info["cluster"] for info in cluster_analysis]
    sizes = [info["train_size"] for info in cluster_analysis]
    val_mses = np.array(
        [info.get("val_mse", info.get("validation_mse", np.nan)) for info in cluster_analysis],
        dtype=float,
    )
    weights = np.array(
        [info.get("global_reliability", info.get("weight", np.nan)) for info in cluster_analysis],
        dtype=float,
    )

    if np.isnan(val_mses).any() or np.isnan(weights).any():
        raise ValueError(
            "cluster_analysis entries must include val_mse/validation_mse and "
            "global_reliability/weight."
        )

    # 1. Cluster sizes
    axes[0, 0].bar(clusters, sizes, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Cluster', fontsize=11)
    axes[0, 0].set_ylabel('Training Samples', fontsize=11)
    axes[0, 0].set_title('Cluster Size Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # 2. Validation MSE
    axes[0, 1].bar(clusters, val_mses, color='coral', edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Cluster', fontsize=11)
    axes[0, 1].set_ylabel('Validation MSE', fontsize=11)
    axes[0, 1].set_title('Validation Error by Cluster', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].set_yscale('log')  # Log scale for better visibility

    # 3. Weights
    max_weight = np.max(weights) if n_clusters > 0 else 0.0
    if max_weight > 0:
        color_scale = weights / max_weight
    else:
        color_scale = np.zeros_like(weights)
    colors = plt.cm.RdYlGn(color_scale)
    axes[1, 0].bar(clusters, weights, color=colors, edgecolor='black', alpha=0.8)
    axes[1, 0].axhline(y=1.0/n_clusters, color='red', linestyle='--',
                      linewidth=2, label='Equal Weight')
    axes[1, 0].set_xlabel('Cluster', fontsize=11)
    axes[1, 0].set_ylabel('Weight', fontsize=11)
    axes[1, 0].set_title('Learned Weights', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # 4. Weight vs Inverse Error relationship
    inv_errors = 1.0 / (np.array(val_mses) + 1e-10)
    inv_errors_norm = inv_errors / inv_errors.sum()

    x = np.arange(n_clusters)
    width = 0.35

    axes[1, 1].bar(x - width/2, weights, width, label='Actual Weight',
                  color='steelblue', alpha=0.7)
    axes[1, 1].bar(x + width/2, inv_errors_norm, width,
                  label='Normalized Inverse Error', color='orange', alpha=0.7)
    axes[1, 1].set_xlabel('Cluster', fontsize=11)
    axes[1, 1].set_ylabel('Value', fontsize=11)
    axes[1, 1].set_title('Weight Derivation Verification',
                        fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].set_xticks(x)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_comprehensive_comparison(results_single, results_mmlr, results_wmmlr,
                                  dataset_name, k, save_path=None):
    """
    Create comprehensive comparison figure for report
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Main comparison metrics
    ax1 = fig.add_subplot(gs[0, :])

    models = ['Single LR', 'MMLR', 'WMMLR']
    rmse_vals = [results_single['RMSE'], results_mmlr['RMSE'], results_wmmlr['RMSE']]
    mae_vals = [results_single['MAE'], results_mmlr['MAE'], results_wmmlr['MAE']]
    r2_vals = [results_single['R2'], results_mmlr['R2'], results_wmmlr['R2']]

    x = np.arange(len(models))
    width = 0.25

    ax1.bar(x - width, rmse_vals, width, label='RMSE', alpha=0.8)
    ax1.bar(x, mae_vals, width, label='MAE', alpha=0.8)
    ax1.bar(x + width, r2_vals, width, label='R2', alpha=0.8)

    ax1.set_ylabel('Metric Value', fontsize=12)
    ax1.set_title(f'Model Comparison - {dataset_name} (K={k})',
                 fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_learning_curves(model, X_train, y_train, X_val, y_val,
                        model_name, save_path=None):
    """
    Plot learning curves to show training progress
    """
    # This requires storing loss history during training
    # Modify your LinearRegression class to store loss_history

    if not hasattr(model, 'loss_history'):
        print("Model doesn't have loss_history attribute")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(model.loss_history, linewidth=2, label='Training Loss')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Loss (MSE + Regularization)', fontsize=12)
    ax.set_title(f'{model_name} - Learning Curve', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_presentation_summary(results_dict, dataset_name, save_path=None):
    """
    Create a single comprehensive figure for presentation slides
    """
    fig = plt.figure(figsize=(18, 10))

    # This would combine multiple visualizations into one figure
    # Customize based on what you want to highlight in presentation

    print("Creating presentation summary figure...")
    # Implementation depends on specific presentation needs

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_cluster_cv_results(df, title):
    """
    df = cluster-level CV dataframe
    """

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # RMSE
    axes[0].bar(df["cluster"], df["cv_rmse_mean"], yerr=df["cv_rmse_std"],
                color="steelblue", alpha=0.7, edgecolor="black")
    axes[0].set_title("Cluster RMSE (CV)")
    axes[0].set_xlabel("Cluster")
    axes[0].set_ylabel("RMSE")

    # MAE
    axes[1].bar(df["cluster"], df["cv_mae_mean"], yerr=df["cv_mae_std"],
                color="orange", alpha=0.7, edgecolor="black")
    axes[1].set_title("Cluster MAE (CV)")
    axes[1].set_xlabel("Cluster")
    axes[1].set_ylabel("MAE")

    # R2
    axes[2].bar(df["cluster"], df["cv_r2_mean"], yerr=df["cv_r2_std"],
                color="green", alpha=0.7, edgecolor="black")
    axes[2].set_title("Cluster R2 (CV)")
    axes[2].set_xlabel("Cluster")
    axes[2].set_ylabel("R2 Score")

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

