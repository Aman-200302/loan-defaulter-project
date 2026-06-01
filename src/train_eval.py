import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)
import xgboost as xgb
import mlflow
import mlflow.sklearn
import joblib
import matplotlib.pyplot as plt
import json, os

# ── 1. Load processed features ────────────────────────────────────
df = pd.read_parquet('data/processed/features.parquet')
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# drop feast helper columns if present
for col in ['event_timestamp', 'applicant_id']:
    if col in X.columns:
        X = X.drop(col, axis=1)

# ── 2. Train / test split ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 3. Train XGBoost ──────────────────────────────────────────────
mlflow.set_experiment("loan-default-prediction")

with mlflow.start_run():
    params = {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "eval_metric": "logloss",
        "random_state": 42
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    # ── 4. Evaluate ───────────────────────────────────────────────
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)

    precision = report['1']['precision']
    recall    = report['1']['recall']
    f1        = report['1']['f1-score']

    # ── 5. Log to MLflow ──────────────────────────────────────────
    mlflow.log_params(params)
    mlflow.log_metrics({"auc": auc, "precision": precision,
                        "recall": recall, "f1": f1})
    mlflow.sklearn.log_model(sk_model=model, name="model")

    # ── 6. Save model ─────────────────────────────────────────────
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/model.pkl')

    # ── 7. Save metrics for CML report ───────────────────────────
    os.makedirs('reports', exist_ok=True)

    metrics = {"auc": round(auc, 4), "precision": round(precision, 4),
               "recall": round(recall, 4), "f1": round(f1, 4)}

    with open('reports/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # ── 8. Confusion matrix plot for CML ─────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['No Default', 'Default'])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False)
    plt.title(f'Confusion Matrix  |  AUC: {auc:.4f}')
    plt.tight_layout()
    plt.savefig('reports/confusion_matrix.png', dpi=100)
    plt.close()

    print("\n── Model Metrics ──────────────────────")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v}")
    print(f"\n  Model saved  → models/model.pkl")
    print(f"  Metrics saved → reports/metrics.json")
    print(f"  Plot saved   → reports/confusion_matrix.png")