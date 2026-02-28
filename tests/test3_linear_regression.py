import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_energy, load_bike, load_airquality
from utils.preprocessing import train_val_test_split
from models.linear_regression import LinearRegressionScratch
from utils.metrics import rmse, mae, r2_score


def run_linear_regression(dataset_name, loader_fn, lambda_reg=0.01):
    print("\n" + "="*80)
    print(f"DATASET: {dataset_name.upper()}")
    print("="*80)

    # Load dataset
    X, y = loader_fn()

    # Split
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)

    # Create model
    model = LinearRegressionScratch(lambda_reg=lambda_reg)
    model.fit(X_train, y_train)

    # Predict
    pred = model.predict(X_test)

    # Print metrics
    print(f"\n--- Results ({dataset_name}) ---")
    print(f"RMSE: {rmse(y_test, pred):.6f}")
    print(f"MAE : {mae(y_test, pred):.6f}")
    print(f"R2  : {r2_score(y_test, pred):.6f}")


if __name__ == "__main__":
    # ENERGY
    run_linear_regression("energy", load_energy, lambda_reg=0.01)

    # BIKE
    run_linear_regression("bike", load_bike, lambda_reg=0.01)

    # AIR QUALITY (slightly higher regularization recommended)
    run_linear_regression("airquality", load_airquality, lambda_reg=0.1)

