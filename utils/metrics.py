import numpy as np
import pandas as pd


# ==============================
# BASIC METRICS
# ==============================
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0


# ==============================
# ADVANCED METRICS
# ==============================
def mape(y_true, y_pred):
    """Mean Absolute Percentage Error (%)"""
    y_true = np.array(y_true, float)
    y_pred = np.array(y_pred, float)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100


def median_ae(y_true, y_pred):
    """Median absolute error"""
    return np.median(np.abs(y_true - y_pred))


def max_error(y_true, y_pred):
    """Worst-case prediction error"""
    return np.max(np.abs(y_true - y_pred))


def adjusted_r2(y_true, y_pred, n_samples, n_features):
    """Adjusted R2 that penalizes model complexity."""
    r2 = r2_score(y_true, y_pred)
    return 1 - ((1 - r2) * (n_samples - 1) / (n_samples - n_features - 1 + 1e-8))


# ===============================================================
# FULL MODEL COMPARISON
# ===============================================================
def compare_models_detailed(y_test, y_pred_single, y_pred_mmlr, y_pred_wmmlr, n_features):
    """
    Compare Single LR, MMLR, WMMLR on all regression metrics.
    """
    n_samples = len(y_test)

    def collect_metrics(pred):
        return {
            "RMSE": rmse(y_test, pred),
            "MAE": mae(y_test, pred),
            "MAPE %": mape(y_test, pred),
            "MedAE": median_ae(y_test, pred),
            "MaxErr": max_error(y_test, pred),
            "R2": r2_score(y_test, pred),
            "Adj R2": adjusted_r2(y_test, pred, n_samples, n_features),
        }

    results = {
        "Single LR": collect_metrics(y_pred_single),
        "MMLR": collect_metrics(y_pred_mmlr),
        "WMMLR": collect_metrics(y_pred_wmmlr),
    }

    # Improvement values
    results["improvements"] = {
        "WMMLR vs Single LR (%)": (
            (results["Single LR"]["RMSE"] - results["WMMLR"]["RMSE"])
            / results["Single LR"]["RMSE"] * 100
        ),
        "WMMLR vs MMLR (%)": (
            (results["MMLR"]["RMSE"] - results["WMMLR"]["RMSE"])
            / results["MMLR"]["RMSE"] * 100
        ),
    }

    return results


# ===============================================================
# TABLE PRINTING
# ===============================================================
def print_results_table(results):
    """
    Pretty console table for all metrics.
    """
    print("\n" + "=" * 95)
    print(
        f"{'Model':<15} {'RMSE':<10} {'MAE':<10} {'MAPE%':<10} {'MedAE':<10} "
        f"{'MaxErr':<10} {'R2':<10} {'Adj R2':<10}"
    )
    print("=" * 95)

    for model in ["Single LR", "MMLR", "WMMLR"]:
        m = results[model]
        print(
            f"{model:<15} "
            f"{m['RMSE']:<10.4f} "
            f"{m['MAE']:<10.4f} "
            f"{m['MAPE %']:<10.2f} "
            f"{m['MedAE']:<10.4f} "
            f"{m['MaxErr']:<10.4f} "
            f"{m['R2']:<10.4f} "
            f"{m['Adj R2']:<10.4f}"
        )

    print("=" * 95)
    print("\nIMPROVEMENTS:")
    for key, val in results["improvements"].items():
        print(f"  {key}: {val:+.2f}%")
    print("=" * 95)


def results_to_dataframe(results):
    """
    Convert results dict to a pandas DataFrame for CSV saving.
    """
    rows = {}
    for model in ["Single LR", "MMLR", "WMMLR"]:
        rows[model] = results[model]

    return pd.DataFrame.from_dict(rows, orient="index")
