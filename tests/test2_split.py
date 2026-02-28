import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_energy
from utils.preprocessing import train_val_test_split

print("Loading Energy dataset...")
X, y = load_energy()
print("Full dataset size:", len(X))

print("\nPerforming split...")
X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)

print("Train size:", len(X_train))
print("Val size:", len(X_val))
print("Test size:", len(X_test))
    