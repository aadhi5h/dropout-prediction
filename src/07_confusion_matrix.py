import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

# --- load features + labels (same as training step) ---
features = pd.read_parquet("data/features_train.parquet")
truth = pd.read_csv("data/train/truth_train.csv", header=None,
                     names=["enrollment_id", "dropout"])
df = features.merge(truth, on="enrollment_id", how="inner")

drop_cols = ["enrollment_id", "first_event_time", "last_event_time",
             "course_id", "to", "dropout"]
X = df.drop(columns=drop_cols).fillna(0)
y = df["dropout"]

# --- same split as training, so results match ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- load the already-trained model instead of retraining ---
model = XGBClassifier()
model.load_model("model_xgb.json")

preds = model.predict(X_test)
cm = confusion_matrix(y_test, preds)

# --- plot ---
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["0 (Retained)", "1 (Dropout)"],
    yticklabels=["0 (Retained)", "1 (Dropout)"],
    cbar=False
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("XGBoost - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Saved confusion_matrix.png")
print(cm)