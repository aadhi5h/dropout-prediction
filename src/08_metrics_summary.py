import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

# --- load features + labels ---
features = pd.read_parquet("data/features_train.parquet")
truth = pd.read_csv("data/train/truth_train.csv", header=None,
                     names=["enrollment_id", "dropout"])
df = features.merge(truth, on="enrollment_id", how="inner")

drop_cols = ["enrollment_id", "first_event_time", "last_event_time",
             "course_id", "to", "dropout"]
X = df.drop(columns=drop_cols).fillna(0)
y = df["dropout"]

# --- same split as training, so results match your original run ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- load the already-trained model instead of retraining ---
model = XGBClassifier()
model.load_model("model_xgb.json")

preds = model.predict(X_test)
preds_proba = model.predict_proba(X_test)[:, 1]

# --- baseline: always predict dropout (class 1) ---
baseline_preds = [1] * len(y_test)


def print_metrics(name, y_true, y_pred, y_proba=None):
    print(f"\n=== {name} ===")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision (class 1): {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall (class 1):    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall (class 0):    {recall_score(y_true, y_pred, pos_label=0, zero_division=0):.4f}")
    print(f"F1-score (class 1):  {f1_score(y_true, y_pred, zero_division=0):.4f}")
    if y_proba is not None:
        print(f"AUC:       {roc_auc_score(y_true, y_proba):.4f}")


print_metrics("XGBoost Model", y_test, preds, preds_proba)
print_metrics("Baseline (always predict dropout)", y_test, baseline_preds)

# --- table-style summary, matching the report's Table 6.1 ---
print("\n=== Summary Table (Model vs Baseline) ===")
print(f"{'Metric':<28}{'Model':>10}{'Baseline':>12}")
print(f"{'AUC':<28}{roc_auc_score(y_test, preds_proba):>10.3f}{'--':>12}")
print(f"{'Accuracy':<28}{accuracy_score(y_test, preds):>10.2%}{accuracy_score(y_test, baseline_preds):>12.2%}")
print(f"{'Recall (Class 0)':<28}{recall_score(y_test, preds, pos_label=0, zero_division=0):>10.2%}"
      f"{recall_score(y_test, baseline_preds, pos_label=0, zero_division=0):>12.2%}")
print(f"{'Recall (Class 1)':<28}{recall_score(y_test, preds, zero_division=0):>10.2%}"
      f"{recall_score(y_test, baseline_preds, zero_division=0):>12.2%}")
