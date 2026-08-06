import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

model = XGBClassifier()
model.load_model("model_xgb.json")

features = pd.read_parquet("data/features_train.parquet")
drop_cols = ["enrollment_id", "first_event_time", "last_event_time", "course_id", "to"]
feature_names = features.drop(columns=drop_cols).columns

importances = model.feature_importances_
imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
imp_df = imp_df.sort_values("importance", ascending=True)

print(imp_df.sort_values("importance", ascending=False))

plt.figure(figsize=(8, 6))
plt.barh(imp_df["feature"], imp_df["importance"])
plt.xlabel("Importance")
plt.title("XGBoost Feature Importance — Dropout Prediction")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
print("Saved feature_importance.png")