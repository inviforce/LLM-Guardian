import pandas as pd
import numpy as np

# ==========================
# Load Dataset
# ==========================
file_path = "/Users/inviforce/Downloads/VS_CODE_C++_PYTHON_JUPYTER_DEV_/LLM Guardian/data/processed/claw_shield_features.csv"

df = pd.read_csv(file_path)

print("="*80)
print("DATASET OVERVIEW")
print("="*80)
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")
print(f"Memory  : {df.memory_usage(deep=True).sum()/1024**2:.2f} MB")
print(f"Duplicate Rows : {df.duplicated().sum()}")

print("\nColumn Names")
for c in df.columns:
    print(c)

# ==========================
# Profile Each Column
# ==========================
profiles = []

for col in df.columns:

    s = df[col]

    profile = {}

    profile["Column"] = col
    profile["Data Type"] = str(s.dtype)

    profile["Total Rows"] = len(s)
    profile["Non Null"] = s.count()
    profile["Null Count"] = s.isna().sum()
    profile["Missing %"] = round(s.isna().mean()*100,2)

    profile["Unique Values"] = s.nunique(dropna=True)

    mode = s.mode(dropna=True)

    if len(mode):
        profile["Most Frequent"] = mode.iloc[0]
        profile["Frequency"] = (s==mode.iloc[0]).sum()
    else:
        profile["Most Frequent"] = None
        profile["Frequency"] = None

    # -----------------------
    # Numeric Columns
    # -----------------------
    if pd.api.types.is_numeric_dtype(s):

        profile["Min"] = s.min()
        profile["25%"] = s.quantile(0.25)
        profile["Median"] = s.median()
        profile["75%"] = s.quantile(0.75)
        profile["Max"] = s.max()

        profile["Mean"] = s.mean()
        profile["Std"] = s.std()
        profile["Variance"] = s.var()

        profile["Range"] = s.max() - s.min()
        profile["IQR"] = s.quantile(.75)-s.quantile(.25)

        profile["Skewness"] = s.skew()
        profile["Kurtosis"] = s.kurt()

        profile["Zeros"] = (s==0).sum()
        profile["Negative Values"] = (s<0).sum()

        profile["Average Text Length"] = None
        profile["Minimum Text Length"] = None
        profile["Maximum Text Length"] = None

    # -----------------------
    # Text Columns
    # -----------------------
    else:

        lengths = s.dropna().astype(str).str.len()

        if len(lengths):

            profile["Average Text Length"] = lengths.mean()
            profile["Minimum Text Length"] = lengths.min()
            profile["Maximum Text Length"] = lengths.max()

        else:

            profile["Average Text Length"] = None
            profile["Minimum Text Length"] = None
            profile["Maximum Text Length"] = None

        profile["Min"] = None
        profile["25%"] = None
        profile["Median"] = None
        profile["75%"] = None
        profile["Max"] = None
        profile["Mean"] = None
        profile["Std"] = None
        profile["Variance"] = None
        profile["Range"] = None
        profile["IQR"] = None
        profile["Skewness"] = None
        profile["Kurtosis"] = None
        profile["Zeros"] = None
        profile["Negative Values"] = None

    # Sample values
    profile["Example Values"] = list(s.dropna().astype(str).head(5))

    profiles.append(profile)

profile_df = pd.DataFrame(profiles)

# ==========================
# Numeric Summary
# ==========================
numeric_summary = df.describe(include=[np.number]).T

# ==========================
# Text Summary
# ==========================
text_summary = df.describe(include=['object']).T

# ==========================
# Missing Values Table
# ==========================
missing = pd.DataFrame({
    "Missing Count": df.isna().sum(),
    "Missing %": df.isna().mean()*100
})

# ==========================
# Save Everything
# ==========================
output_file = "claw_shield_dataset_profile.xlsx"

profile_df.to_csv("complete_profile.csv", index=False)

numeric_summary.to_csv("numeric_summary.csv")

text_summary.to_csv("text_summary.csv")

missing.to_csv("missing_values.csv")

print("\n" + "="*80)
print("PROFILE CREATED SUCCESSFULLY")
print("="*80)
print("Files Saved:")
print("1. complete_profile.csv")
print("2. numeric_summary.csv")
print("3. text_summary.csv")
print("4. missing_values.csv")