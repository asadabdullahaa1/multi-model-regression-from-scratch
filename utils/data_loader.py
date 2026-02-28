import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ================================================================
# 1. ENERGY DATASET
# ================================================================
def load_energy(path="data/ENB2012_data.csv", target="Y1"):
    """Load Energy Efficiency dataset (handles missing headers + scaling)."""

    try:
        df = pd.read_csv(path)
    except:
        df = pd.read_csv(path, header=None)
        df.columns = ["X1","X2","X3","X4","X5","X6","X7","X8","Y1","Y2"]

    # Determine feature/target columns
    if all(c in df.columns for c in ["X1","X2","X3","X4","X5","X6","X7","X8"]):
        feature_cols = ["X1","X2","X3","X4","X5","X6","X7","X8"]
        target_col = target
    else:
        feature_cols = df.columns[:8]
        target_col = df.columns[8] if target == "Y1" else df.columns[9]

    X = df[feature_cols].astype(float).values
    y = df[target_col].astype(float).values

    # Remove bad rows
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]

    print(f"Energy dataset: {len(X)} samples after cleaning.")

    # Scale
    X = StandardScaler().fit_transform(X)

    return X, y


# ================================================================
# 2. BIKE SHARING DATASET
# ================================================================
def load_bike(path="data/hour.csv"):
    """Load Bike Sharing dataset with leakage columns removed + scaling."""

    df = pd.read_csv(path)

    # Remove leakage
    for col in ["instant", "dteday", "casual", "registered"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Identify features
    feature_cols = [c for c in df.columns if c not in ["cnt", "count"]]
    target_col = "cnt" if "cnt" in df.columns else ("count" if "count" in df.columns else df.columns[-1])

    X = df[feature_cols].astype(float).values
    y = df[target_col].astype(float).values

    # Remove invalid rows
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]

    print(f"Bike dataset: {len(X)} samples, {X.shape[1]} features.")

    # Scale
    X = StandardScaler().fit_transform(X)

    return X, y


def load_airquality(path="data/AirQualityUCI.csv", target="CO(GT)"):
    """
    Stable AirQuality loader:
      - replaces -200 with NaN
      - fills NaN using median (no chained assignment warning)
      - keeps all important numeric pollutant + sensor + environment features
      - excludes Date, Time
      - scales X
    """

    df = pd.read_csv(path)

    # Replace -200 with NaN
    df = df.replace(-200.0, np.nan)

    # Define columns
    gt_cols = ["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]
    sensor_cols = [
        "PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)",
        "PT08.S4(NO2)", "PT08.S5(O3)"
    ]
    env_cols = ["T", "RH", "AH"]

    all_features = gt_cols + sensor_cols + env_cols

    # Ensure columns exist
    for col in all_features:
        if col not in df.columns:
            raise ValueError(f"Missing expected AirQuality column: {col}")

    # ---- FIX: safe median imputation (NO CHAINED ASSIGNMENT) ----
    df[all_features] = df[all_features].fillna(df[all_features].median())

    # Extract y
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in AirQuality dataset.")

    y = df[target].values.astype(float)

    # Remove target from X
    feature_cols = [c for c in all_features if c != target]
    X = df[feature_cols].values.astype(float)

    print(f"AirQuality stable load: {X.shape[0]} samples, {X.shape[1]} features.")
    print(f"Target variable: {target}")

    # Scale X
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y
