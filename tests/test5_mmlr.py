import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_energy
from utils.preprocessing import train_val_test_split
from models.mmlr_model import MMLR
from utils.metrics import rmse, mae, r2_score
import numpy as np

print("Loading Energy dataset...")
X, y = load_energy()

print("Splitting...")
X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)

print("\nTraining MMLR with k=3...")
mmlr = MMLR(k=3)
mmlr.fit(X_train, y_train)

print("\nPredicting on test set...")
pred = mmlr.predict(X_test)

print("\nMMLR Results:")
print("  RMSE:", rmse(y_test, pred))
print("  MAE:", mae(y_test, pred))
print("  R2:", r2_score(y_test, pred))

print("\nCluster sizes:", mmlr.cluster_info)

