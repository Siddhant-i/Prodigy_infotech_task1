
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

Housing_data = pd.read_csv("train.csv")
Housing_data["TotalBath"] = Housing_data["FullBath"] + (0.5 * Housing_data["HalfBath"])

house_features  = ["GrLivArea", "BedroomAbvGr", "TotalBath"]
target_price    = "SalePrice"
housing_model  = Housing_data[house_features + [target_price]].dropna()

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor"  : "#f9f9f9",
    "axes.grid"       : True,
    "grid.alpha"      : 0.4,
    "font.size"       : 11,
})
plt.figure(figsize=(8, 4))
plt.hist(housing_model["SalePrice"], bins=50, color="#4C72B0", edgecolor="white", alpha=0.85)
plt.axvline(housing_model["SalePrice"].mean(), color="red",   linestyle="--", linewidth=1.5, label=f"Mean: ${housing_model['SalePrice'].mean():,.0f}")
plt.axvline(housing_model["SalePrice"].median(), color="orange", linestyle="--", linewidth=1.5, label=f"Median: ${housing_model['SalePrice'].median():,.0f}")
plt.xlabel("Sale Price ($)")
plt.ylabel("Number of Houses")
plt.title("Distribution of House Sale Prices")
plt.legend()
plt.tight_layout()
plt.savefig("chart1_price_distribution.png", dpi=150)
plt.show()
print("chart1_price_distribution.png")

plt.figure(figsize=(8, 5))
plt.scatter(housing_model["GrLivArea"], housing_model["SalePrice"],
            alpha=0.4, color="#2196F3", edgecolors="none", s=25)
plt.xlabel("Living Area (sq ft)")
plt.ylabel("Sale Price ($)")
plt.title("Square Footage vs Sale Price")
plt.tight_layout()
plt.savefig("chart2_sqft_vs_price.png", dpi=150)
plt.show()
print("chart2_sqft_vs_price.png")

bedroom_avg = housing_model.groupby("BedroomAbvGr")["SalePrice"].mean().reset_index()

plt.figure(figsize=(8, 4))
bars = plt.bar(bedroom_avg["BedroomAbvGr"].astype(str),
               bedroom_avg["SalePrice"],
               color="#66BB6A", edgecolor="white", alpha=0.9)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height + 2000,
             f"${height:,.0f}", ha="center", va="bottom", fontsize=9)
plt.xlabel("Number of Bedrooms")
plt.ylabel("Average Sale Price ($)")
plt.title("Average Sale Price by Number of Bedrooms")
plt.tight_layout()
plt.savefig("chart3_bedrooms_vs_price.png", dpi=150)
plt.show()
print(" chart3_bedrooms_vs_price.png")

plt.figure(figsize=(6, 4))
corr = housing_model.corr()        
sns.heatmap(
    corr, annot=True, fmt=".2f", cmap="RdYlGn",
    linewidths=0.5, linecolor="white",
    annot_kws={"size": 11}, cbar_kws={"shrink": 0.8}
)
plt.title("Correlation Between Features and Price")
plt.tight_layout()
plt.savefig("chart4_correlation_heatmap.png", dpi=150)
plt.show()
print("chart4_correlation_heatmap.png")

print("\n" + "=" * 55)
print("  KEY OBSERVATIONS FOR YOUR REPORT")
print("=" * 55)
corr_with_price = housing_model.corr()["SalePrice"].drop("SalePrice")
for feature_name, correlation_value in corr_with_price.items():
    print(f"  {feature_name:20s}  correlation with price = {correlation_value:.3f}")
print("=" * 55)
print("\n visualize okay, Run processing one")
