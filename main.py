"""
MAIN EXPERIMENT SCRIPT (AUTO-SAVING VERSION)
Creates a timestamped results folder and saves ALL plots non-interactively.
"""

import os
import sys
import time

import numpy as np

from utils.data_loader import load_energy, load_bike, load_airquality
from utils.preprocessing import train_val_test_split
from utils.metrics import compare_models_detailed, print_results_table, rmse, mae, r2_score

from models.mmlr_model import MMLR
from models.wmmlr_model import WMMLR
from models.linear_regression import LinearRegressionScratch


import matplotlib

from utils.visualizations import plot_clusters, plot_model_weights, plot_predictions_vs_actual
if not hasattr(sys, 'ps1'):   # running as script, not notebook
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd

'''
# Import your existing modules
from dataset_loaders import load_energy, load_bike, load_airquality
from metric_evaluations import (rmse, mae, r2_score, mape, median_ae, max_error,
                               adjusted_r2, compare_models_detailed, print_results_table,
                               results_to_dataframe, statistical_significance_test)
from pre_processing import train_val_test_split
from visualization import (plot_clusters, plot_model_weights, compare_model_errors,
                          plot_predictions_vs_actual, plot_residuals, plot_cluster_performance,
                          plot_comprehensive_comparison)
from kmeans_scratch import KMeansScratch
from linear_regression_scratch import LinearRegressionScratch
from mmlr import MMLR
from wmmlr import WMMLR
 '''

# -------------------------------------------------------------------
# Helper: Create timestamp folder
# -------------------------------------------------------------------

def create_results_folders():
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    root = f"results_run_{timestamp}"
    os.makedirs(root, exist_ok=True)

    subfolders = {
        "energy": os.path.join(root, "energy"),
        "bike": os.path.join(root, "bike"),
        "airquality": os.path.join(root, "airquality")
    }

    for folder in subfolders.values():
        os.makedirs(folder, exist_ok=True)

    return root, subfolders


# -------------------------------------------------------------------
# Plot helper: Auto-save
# -------------------------------------------------------------------

def save_plot(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


# ================================================================
# EXPERIMENTS 
# ================================================================

def run_comprehensive_experiment(dataset_name, out_dir, k_values=(3,5,7)):
    print("\n" + "="*80)
    print(f"DATASET: {dataset_name.upper()}")
    print("="*80)

    # LOAD DATA
    if dataset_name == "energy":
        X, y = load_energy()
    elif dataset_name == "bike":
        X, y = load_bike()
    else:
        X, y = load_airquality()

    # SPLIT - Use consistent split for all models
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(
        X, y, train_ratio=0.6, val_ratio=0.2
    )

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # -------------------------------------------------------------------
    # FIXED: Use consistent training approach
    # -------------------------------------------------------------------

    # For Single LR and MMLR: Use train+val for final training (as in paper)
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.hstack([y_train, y_val])

    # For WMMLR: Keep original split to use validation for reliability weights

    # -------------------------------------------------------------------
    # BASELINE: Linear Regression
    # -------------------------------------------------------------------

    baseline = LinearRegressionScratch(lambda_reg=0.01)  # Use closed-form
    baseline.fit(X_train_full, y_train_full)
    y_pred_single = baseline.predict(X_test)

    baseline_results = {
        "RMSE": rmse(y_test, y_pred_single),
        "MAE": mae(y_test, y_pred_single),
        "R2": r2_score(y_test, y_pred_single)
    }

    print(f"\nSingle LR -> RMSE={baseline_results['RMSE']:.4f}")

    # Store results for all k values
    all_results = {}

    # -------------------------------------------------------------------
    # MMLR + WMMLR for each K
    # -------------------------------------------------------------------

    for k in k_values:
        print("\n" + "-"*60)
        print(f"K = {k}")
        print("-"*60)

        # -------------------------
        # MMLR - Use same training data as baseline
        # -------------------------

        mmlr = MMLR(k=k, lambda_reg=0.01, random_state=42)
        mmlr.fit(X_train_full, y_train_full)
        y_pred_mmlr = mmlr.predict(X_test)

        mmlr_results = {
            "RMSE": rmse(y_test, y_pred_mmlr),
            "MAE": mae(y_test, y_pred_mmlr),
            "R2": r2_score(y_test, y_pred_mmlr)
        }

        print(f"MMLR     -> RMSE={mmlr_results['RMSE']:.4f}")

        # -------------------------
        # WMMLR - Use original split for validation-based reliability
        # -------------------------

        wmmlr = WMMLR(k=k, lambda_reg=0.01, random_state=42)
        wmmlr.fit(X_train, y_train, X_val, y_val)
        y_pred_wmmlr = wmmlr.predict(X_test)

        wmmlr_results = {
            "RMSE": rmse(y_test, y_pred_wmmlr),
            "MAE": mae(y_test, y_pred_wmmlr),
            "R2": r2_score(y_test, y_pred_wmmlr)
        }

        print(f"WMMLR    -> RMSE={wmmlr_results['RMSE']:.4f}")

        # CHECK CLUSTER QUALITY
        # After WMMLR training, add cluster analysis
        print(f"\nWMMLR Cluster Analysis for K={k}:")
        for c in range(k):
            info = wmmlr.cluster_info[c]
            val_mse = info.get("val_mse")
            reliability = info.get("global_reliability")
            val_mse_str = f"{val_mse:.4f}" if val_mse is not None else "N/A"
            reliability_str = f"{reliability:.4f}" if reliability is not None else "N/A"
            print(
                f"  Cluster {c}: size={info['train_size']}, "
                f"val_mse={val_mse_str}, reliability={reliability_str}"
            )

        # Check if any clusters are problematic
        tiny_clusters = [c for c in range(k) if wmmlr.cluster_info[c]['train_size'] < 10]
        if tiny_clusters:
            print(f"  WARNING: Tiny clusters detected: {tiny_clusters}")

        # -------------------------------------------------------------------
        # COMPREHENSIVE EVALUATION
        # -------------------------------------------------------------------
        n_features = X_train.shape[1]
        results = compare_models_detailed(
            y_test, y_pred_single, y_pred_mmlr, y_pred_wmmlr, n_features
        )

        all_results[k] = results

        # Print detailed results table
        print_results_table(results)

        # -------------------------------------------------------------------
        # SAVE PLOTS for this K value
        # -------------------------------------------------------------------

        # 1. Cluster visualization (using MMLR clusters since they use full data)
        try:
            plot_clusters(X_train_full, mmlr.cluster_model.labels,
                         title=f"{dataset_name} - Clusters (K={k})",
                         save_path=os.path.join(out_dir, f"clusters_k{k}.png"))
        except Exception as e:
            print(f"Could not plot clusters: {e}")

        # 2. WMMLR cluster weights and analysis
        try:
            cluster_sizes = [wmmlr.cluster_info[c]['train_size'] for c in range(k)]
            plot_model_weights(wmmlr.global_reliability, cluster_sizes,
                             title=f"{dataset_name} - WMMLR Weights (K={k})",
                             save_path=os.path.join(out_dir, f"weights_k{k}.png"))
        except Exception as e:
            print(f"Could not plot weights: {e}")

        # 3. Predictions vs actual for all models
        plot_predictions_vs_actual(y_test, y_pred_single, "Single LR", f"{dataset_name} K={k}",
                                 save_path=os.path.join(out_dir, f"pred_single_k{k}.png"))

        plot_predictions_vs_actual(y_test, y_pred_mmlr, "MMLR", f"{dataset_name} K={k}",
                                 save_path=os.path.join(out_dir, f"pred_mmlr_k{k}.png"))

        plot_predictions_vs_actual(y_test, y_pred_wmmlr, "WMMLR", f"{dataset_name} K={k}",
                                 save_path=os.path.join(out_dir, f"pred_wmmlr_k{k}.png"))

        # 4. RMSE comparison bar chart
        plt.figure(figsize=(8, 6))
        models = ["Single LR", "MMLR", "WMMLR"]
        rmse_values = [results["Single LR"]["RMSE"],
                      results["MMLR"]["RMSE"],
                      results["WMMLR"]["RMSE"]]

        bars = plt.bar(models, rmse_values, color=['lightblue', 'lightgreen', 'lightcoral'])
        plt.ylabel("RMSE")
        plt.title(f"RMSE Comparison - {dataset_name} (K={k})")

        # Add value labels on bars
        for bar, value in zip(bars, rmse_values):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{value:.4f}', ha='center', va='bottom')

        save_plot(os.path.join(out_dir, f"rmse_compare_k{k}.png"))

        # 5. Save cluster analysis for WMMLR
        try:
            cluster_analysis = wmmlr.get_cluster_analysis()
            cluster_df = pd.DataFrame(cluster_analysis)
            cluster_df.to_csv(os.path.join(out_dir, f"cluster_analysis_k{k}.csv"), index=False)
        except Exception as e:
            print(f"Could not save cluster analysis: {e}")

    # -------------------------------------------------------------------
    # SUMMARY ACROSS ALL K VALUES
    # -------------------------------------------------------------------

    print("\n" + "="*80)
    print(f"SUMMARY FOR {dataset_name.upper()} ACROSS K VALUES")
    print("="*80)

    summary_data = []
    for k in k_values:
        results = all_results[k]
        summary_data.append({
            'K': k,
            'Single_LR_RMSE': results["Single LR"]["RMSE"],
            'MMLR_RMSE': results["MMLR"]["RMSE"],
            'WMMLR_RMSE': results["WMMLR"]["RMSE"],
            'Improvement_vs_Single': results["improvements"]["WMMLR vs Single LR (%)"],
            'Improvement_vs_MMLR': results["improvements"]["WMMLR vs MMLR (%)"]
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    # Save summary
    summary_df.to_csv(os.path.join(out_dir, "summary_across_k.csv"), index=False)

    # Plot performance across K values
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, summary_df['Single_LR_RMSE'], marker='o', label='Single LR', linewidth=2)
    plt.plot(k_values, summary_df['MMLR_RMSE'], marker='s', label='MMLR', linewidth=2)
    plt.plot(k_values, summary_df['WMMLR_RMSE'], marker='^', label='WMMLR', linewidth=2)
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('RMSE')
    plt.title(f'Model Performance vs Cluster Count - {dataset_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_plot(os.path.join(out_dir, "performance_vs_k.png"))

    print(f"\nAll results and plots saved to: {out_dir}")

    return all_results


# ================================================================
# QUICK VALIDATION FUNCTION
# ================================================================

def quick_validation():
    """Run a quick validation to check if fixes work"""
    print("Running quick validation...")

    # Use a small dataset for quick test
    X, y = load_energy()
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)

    # Test Single LR
    baseline = LinearRegressionScratch(lambda_reg=0.01)
    baseline.fit(np.vstack([X_train, X_val]), np.hstack([y_train, y_val]))
    y_pred = baseline.predict(X_test)

    print(f"Quick validation - Single LR RMSE: {rmse(y_test, y_pred):.4f}")
    print("If this runs without errors, your fixes are working!")

    return rmse(y_test, y_pred)


# ================================================================
# MAIN
# ================================================================

def main():
    print("="*80)
    print("MMLR / WMMLR EXPERIMENTS ")
    print("="*80)

    # First run quick validation
    try:
        quick_validation()
        print("[OK] Quick validation passed!")
    except Exception as e:
        print(f"[ERROR] Quick validation failed: {e}")
        return

    # Create results folders
    root, folders = create_results_folders()
    print(f"Results will be saved to: {root}")

    # Run experiments for each dataset
    for dataset in ["energy", "bike", "airquality"]:
        try:
            print(f"\n{'='*80}")
            print(f"STARTING EXPERIMENT: {dataset.upper()}")
            print(f"{'='*80}")

            results = run_comprehensive_experiment(dataset, folders[dataset], k_values=(3, 5, 7))

            print(f"\n[OK] Completed {dataset.upper()} successfully!")

        except Exception as e:
            print(f"\n[ERROR] Error on {dataset}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED")
    print(f"Results saved to: {root}")
    print("="*80)


if __name__ == "__main__":
    main()
