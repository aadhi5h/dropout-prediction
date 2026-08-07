import pandas as pd

features = pd.read_parquet("data/features_train.parquet")
truth = pd.read_csv("data/train/truth_train.csv", header=None,
                     names=["enrollment_id", "dropout"])

df = features.merge(truth, on="enrollment_id", how="inner")

# add a readable label for dashboard filters
df["dropout_label"] = df["dropout"].map({1: "Dropped Out", 0: "Retained"})

# drop timestamp columns Tableau doesn't need in raw form (keep derived ones)
df.to_csv("dashboard_export.csv", index=False)
print("Saved dashboard_export.csv:", df.shape)