

import pandas as pd
import joblib

housedata_model  = joblib.load("house_price_model.pkl")
price_scaler = joblib.load("scaler.pkl")

House_features = ["GrLivArea", "BedroomAbvGr", "TotalBath"]

def predict_house_price(sqft, bedrooms, bathrooms):
    input_housedata = pd.DataFrame([[sqft, bedrooms, bathrooms]], columns=House_features)
    input_scaled = price_scaler.transform(input_housedata)
    predicted = housedata_model.predict(input_scaled)[0]
    return max(predicted, 0)   # price can't be negative


test_houses = [
    {"sqft": 800,  "bedrooms": 2, "bathrooms": 1.0, "label": "Small starter home"},
    {"sqft": 1200, "bedrooms": 3, "bathrooms": 1.5, "label": "Average 3-bed home"},
    {"sqft": 1800, "bedrooms": 3, "bathrooms": 2.0, "label": "Comfortable family home"},
    {"sqft": 2500, "bedrooms": 4, "bathrooms": 2.5, "label": "Large 4-bed home"},
    {"sqft": 4000, "bedrooms": 5, "bathrooms": 3.5, "label": "Luxury home"},
]

print("=" * 65)
print("  HOUSE PRICE PREDICTIONS")
print("=" * 65)
print(f"\n  {'House Type':<28} {'Sq Ft':>6} {'Beds':>5} {'Baths':>6} {'Predicted Price':>15}")
print(f"  {'-'*62}")

for h in test_houses:
    price = predict_house_price(h["sqft"], h["bedrooms"], h["bathrooms"])
    print(f"  {h['label']:<28} {h['sqft']:>6}  {h['bedrooms']:>4}  {h['bathrooms']:>5.1f}  ${price:>13,.0f}")

print("=" * 65)


print("\n--- Predict YOUR house ---")
print("Enter house details below:")

try:
    sqft = float(input("  Square footage   : "))
    bedrooms  = int(input("  Bedrooms         : "))
    full_bath = int(input("  Full bathrooms   : "))
    half_bath = int(input("  Half bathrooms   : "))
    bathrooms = full_bath + 0.5 * half_bath

    price = predict_house_price(sqft, bedrooms, bathrooms)
    print(f"\n  Predicted Sale Price: ${price:,.0f}")
    print(f"     (Range estimate: ${price*0.90:,.0f} – ${price*1.10:,.0f})")

except (ValueError, KeyboardInterrupt):
    print("\n  (Skipped interactive input — running in non-interactive mode)")

print("\n data_prediction okay.. Project completed")
