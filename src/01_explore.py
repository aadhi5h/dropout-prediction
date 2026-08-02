import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)

# --- enrollment ---
enrollment = pd.read_csv("data/train/enrollment_train.csv")
print("=== enrollment_train ===")
print(enrollment.shape)
print(enrollment.head())
print(enrollment.dtypes)
print("Unique students:", enrollment["username"].nunique())
print("Unique courses:", enrollment["course_id"].nunique())
print()

# --- truth (labels) ---
truth = pd.read_csv("data/train/truth_train.csv", header=None,
                     names=["enrollment_id", "dropout"])
print("=== truth_train ===")
print(truth.shape)
print(truth["dropout"].value_counts(normalize=True))
print()

# --- object (course structure) ---
obj = pd.read_csv("data/object.csv")
print("=== object ===")
print(obj.shape)
print(obj.head())
print(obj["category"].value_counts() if "category" in obj.columns else obj.columns)
print()

# --- date (course start/end) ---
date = pd.read_csv("data/date.csv")
print("=== date ===")
print(date.shape)
print(date.head())