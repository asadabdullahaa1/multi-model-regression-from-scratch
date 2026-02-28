import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_energy
from utils.preprocessing import train_val_test_split
from models.wmmlr_model import WMMLR
from utils.metrics import rmse, mae, r2_score

print("Loading Energy dataset...")
X, y = load_energy()

print("Splitting...")
X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)

print("\nTraining WMMLR with k=3...")
wmmlr = WMMLR(k=3)
wmmlr.fit(X_train, y_train, X_val, y_val)

print("\nPredicting on test set...")
pred = wmmlr.predict(X_test)

print("\nWMMLR Results:")
print("  RMSE:", rmse(y_test, pred))
print("  MAE:", mae(y_test, pred))
print("  R2: ", r2_score(y_test, pred))

print("\nCluster Validation MSE:")
print(wmmlr.val_mse)

print("\nGlobal reliability (inverse MSE normalized):")
print(wmmlr.global_reliability)

print("\nCluster Analysis:")
for info in wmmlr.get_cluster_analysis():
    print(info)

