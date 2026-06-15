# Kaggle  dataset CSV file read and open,load 

import pandas as pd

House_dataset = pd.read_csv("train.csv")

print(f"=" * 55)
print(f"  DATASET OVERVIEW")
print(f"=" * 55)
print(f"  Total rows    : {House_dataset.shape[0]}")
print(f"  Total columns : {House_dataset.shape[1]}")
print(f"=" * 55)

print(f"\n First 5 rows of the full dataset")
print(House_dataset.head())

#   GrLivArea    = Above-ground living area (sqft)
#   BedroomAbvGr = No of bedrooms
#   FullBath     = Full bathrooms
#   HalfBath     = Half bathrooms (0.5)
#   SalePrice    = Target variable (prediction work rahega)
House_dataset["TotalBath"] = House_dataset["FullBath"] + (0.5 * House_dataset["HalfBath"])

features = ["GrLivArea", "BedroomAbvGr", "TotalBath"]
target   = "SalePrice"

HDS_model = House_dataset[features + [target]].dropna()

print("\nOur working dataframe (selected features)")
print(HDS_model.head(10))

print("\n Statistical Summary")
print(HDS_model.describe().round(2))

print("\n Missing Values in Selected Columns")
print(HDS_model.isnull().sum())

print("\n Data Types ")
print(HDS_model.dtypes)

print("\n Data loading done now, after this Run data_visulize")
