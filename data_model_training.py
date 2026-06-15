
import joblib
from sklearn.linear_model import LinearRegression

# X = input , Y =  Output
X_train_scaled, X_test_scaled, y_train, y_test = joblib.load("processed_data.pkl")

house_features = ["GrLivArea", "BedroomAbvGr", "TotalBath"]

housedata_model = LinearRegression()
housedata_model.fit(X_train_scaled, y_train)

print(f"=" * 55)
print(f"  MODEL TRAINED SUCCESSFULLY")
print(f"=" * 55)

# Coefficients
#   Positive coefficient = feature increases price
#   Negative coefficient = feature decreases price
#   Larger absolute value = stronger influence on price

print(f"\n Learned Coefficients (after scaling)")
print(f"  {'Feature':<20} Coefficient")
print(f"  {'-'*40}")

for name, coef in zip(house_features, housedata_model.coef_):
    direction = " increases price" if coef > 0 else "decreases price"
    print(f"  {name:<20} {coef:+.2f}   ({direction})")
print(f"\n  Intercept (bias) : {housedata_model.intercept_:,.2f}")

joblib.dump(housedata_model, "house_price_model.pkl")
print(f" Saved: house_price_model.pkl")

print("Data modeling done now, evaluation part")
