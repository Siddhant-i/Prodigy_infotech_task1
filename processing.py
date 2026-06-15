
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

h_data = pd.read_csv("train.csv")

h_data["TotalBath"] = h_data["FullBath"] + (0.5 * h_data["HalfBath"])

house_features = ["GrLivArea", "BedroomAbvGr", "TotalBath"]
target   = "SalePrice"

process_data = h_data[house_features + [target]].dropna()

X = process_data[house_features]
y = process_data[target]

print("=" * 55)
print("  PREPROCESSING SUMMARY")
print("=" * 55)
print(f"  Total usable samples : {len(X)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"  Training samples     : {len(X_train)}  (80%)")
print(f"  Testing  samples     : {len(X_test)}   (20%)")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

for name, mean, std in zip(house_features, scaler.mean_, scaler.scale_):
    print(f"  {name:20s}  mean={mean:8.2f}  std={std:8.2f}")

print(pd.DataFrame(X_train_scaled[:5], columns=house_features).round(3))

joblib.dump(scaler,  "scaler.pkl")
joblib.dump((X_train_scaled, X_test_scaled, y_train, y_test), "processed_data.pkl")
