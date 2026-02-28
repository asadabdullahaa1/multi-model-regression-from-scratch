import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_energy, load_bike, load_airquality

print("Testing Energy dataset...")
X, y = load_energy()
print(X.shape, y.shape)

print("\nTesting Bike dataset...")
X2, y2 = load_bike()
print(X2.shape, y2.shape)

print("\nTesting Air Quality dataset...")
X3, y3 = load_airquality()
print(X3.shape, y3.shape)
