import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# --- load ---
features = pd.read_parquet("data/features_train.parquet")
truth = pd.read_csv("data/train/truth_train.csv", header=None,
                     names=["enrollment_id", "dropout"])

df = features.merge(truth, on="enrollment_id", how="inner")
print("Merged shape:", df.shape)
print("Nulls per column:\n", df.isnull().sum())

# drop non-feature columns before training
drop_cols = ["enrollment_id", "first_event_time", "last_event_time",
             "course_id", "to", "dropout"]
X = df.drop(columns=drop_cols)
y = df["dropout"]

X = X.fillna(0)  # e.g. events_per_active_day edge cases, missing date joins

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# handle class imbalance (79% dropout, 21% not)
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    eval_metric="auc",
    random_state=42,
)
model.fit(X_train, y_train)

preds_proba = model.predict_proba(X_test)[:, 1]
preds = model.predict(X_test)

print("\n=== AUC ===")
print(roc_auc_score(y_test, preds_proba))

print("\n=== classification report ===")
print(classification_report(y_test, preds))

print("\n=== confusion matrix ===")
print(confusion_matrix(y_test, preds))

# --- baseline comparison: always predict dropout ---
baseline_preds = [1] * len(y_test)
print("\n=== baseline (always predict dropout=1) ===")
print(classification_report(y_test, baseline_preds))

model.save_model("model_xgb.json")
print("\nModel saved to model_xgb.json")