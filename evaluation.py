

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

house_price_model  = joblib.load("house_price_model.pkl")
X_train_scaled, X_test_scaled, y_train, y_test = joblib.load("processed_data.pkl")

y_pred_test  = house_price_model.predict(X_test_scaled)
y_pred_train = house_price_model.predict(X_train_scaled)

mae_test  = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test   = r2_score(y_test, y_pred_test)

mae_train  = mean_absolute_error(y_train, y_pred_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
r2_train   = r2_score(y_train, y_pred_train)

print("=" * 55)
print("  MODEL EVALUATION RESULTS")
print("=" * 55)
print(f"\n  {'Metric':<10}  {'Train':>12}  {'Test':>12}")
print(f"  {'-'*38}")
print(f"  {'MAE':<10}  ${mae_train:>10,.0f}  ${mae_test:>10,.0f}")
print(f"  {'RMSE':<10}  ${rmse_train:>10,.0f}  ${rmse_test:>10,.0f}")
print(f"  {'R²':<10}  {r2_train:>12.4f}  {r2_test:>12.4f}")
print("=" * 55)

print(f"\nModel Test R^2 Score : {r2_test:.4f}")

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "#f9f9f9",
                      "axes.grid": True, "grid.alpha": 0.4, "font.size": 11})

plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred_test, alpha=0.45, color="#E53935", edgecolors="none", s=30, label="Predictions")
min_val = min(y_test.min(), y_pred_test.min())
max_val = max(y_test.max(), y_pred_test.max())
plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=1.5, label="Perfect prediction line")
plt.xlabel("Actual Sale Price ($)")
plt.ylabel("Predicted Sale Price ($)")
plt.title(f"Actual vs Predicted Prices  (R² = {r2_test:.3f})")
plt.legend()
plt.tight_layout()
plt.savefig("chart5_actual_vs_predicted.png", dpi=150)
plt.show()

residuals = y_test - y_pred_test

plt.figure(figsize=(7, 4))
plt.scatter(y_pred_test, residuals, alpha=0.4, color="#7B1FA2", edgecolors="none", s=25)
plt.axhline(0, color="black", linewidth=1.2, linestyle="--")
plt.xlabel("Predicted Price ($)")
plt.ylabel("Residual (Actual − Predicted) ($)")
plt.title("Residuals Plot")
plt.tight_layout()
plt.savefig("chart6_residuals.png", dpi=150)
plt.show()

predict_results = pd.DataFrame({
    "Actual Price"   : y_test.values[:10],
    "Predicted Price": y_pred_test[:10].round(0),
    "Error ($)"      : (y_test.values[:10] - y_pred_test[:10]).round(0),
})

print(predict_results.to_string(index=False))

print("Model evaluation completed")
