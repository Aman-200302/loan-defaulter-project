import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)
import xgboost as xgb
import mlflow
import mlflow.xgboost
from mlflow.models.signature import infer_signature
import optuna
import joblib
import matplotlib.pyplot as plt
import json, os, warnings
warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────
EXPERIMENT_NAME = "loan-default-prediction"
MODEL_NAME      = "loan-default-classifier"
MLFLOW_URI      = "mlruns"          # local folder — MLflow UI reads this
N_TRIALS        = 20                # number of Optuna trials

# ── 1. Load data ──────────────────────────────────────────────────
df = pd.read_parquet('data/processed/features.parquet')
X  = df.drop(columns=['loan_status', 'event_timestamp',
                       'applicant_id'], errors='ignore')
y  = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 2. MLflow setup ───────────────────────────────────────────────
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# ── 3. Optuna objective — each trial is one MLflow run ────────────
def objective(trial):
    params = {
        "n_estimators":   trial.suggest_int("n_estimators", 100, 500),
        "max_depth":      trial.suggest_int("max_depth", 3, 8),
        "learning_rate":  trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":      trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "eval_metric":    "logloss",
        "random_state":   42,
        "use_label_encoder": False,
    }

    with mlflow.start_run(run_name=f"trial_{trial.number}"):
        # log hyperparams
        mlflow.log_params(params)
        mlflow.log_param("trial_number", trial.number)

        # train
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)

        # evaluate
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        auc       = roc_auc_score(y_test, y_proba)
        report    = classification_report(y_test, y_pred, output_dict=True)
        precision = report['1']['precision']
        recall    = report['1']['recall']
        f1        = report['1']['f1-score']

        # log metrics
        mlflow.log_metrics({
            "auc": auc, "precision": precision,
            "recall": recall, "f1": f1
        })

        # log model with input signature
        signature = infer_signature(X_train, y_pred)
        mlflow.xgboost.log_model(model, "model", signature=signature)

    return auc   # Optuna maximises this

# ── 4. Run Optuna study ───────────────────────────────────────────
print(f"\nRunning {N_TRIALS} Optuna trials — each logged to MLflow...\n")
study = optuna.create_study(direction="maximize",
                            study_name=EXPERIMENT_NAME)
optuna.logging.set_verbosity(optuna.logging.WARNING)
study.optimize(objective, n_trials=N_TRIALS)

# ── 5. Retrain best model on full data ───────────────────────────
print(f"\nBest trial: {study.best_trial.number}")
print(f"Best AUC:   {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

best_params = {**study.best_params,
               "eval_metric": "logloss",
               "random_state": 42,
               "use_label_encoder": False}

with mlflow.start_run(run_name="best_model_final"):
    mlflow.log_params(best_params)
    mlflow.log_param("selected_by", "optuna_maximize_auc")

    best_model = xgb.XGBClassifier(**best_params)
    best_model.fit(X_train, y_train,
                   eval_set=[(X_test, y_test)],
                   verbose=False)

    y_pred  = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    auc       = roc_auc_score(y_test, y_proba)
    report    = classification_report(y_test, y_pred, output_dict=True)
    precision = report['1']['precision']
    recall    = report['1']['recall']
    f1        = report['1']['f1-score']

    mlflow.log_metrics({
        "auc": auc, "precision": precision,
        "recall": recall, "f1": f1
    })

    # ── 6. Confusion matrix ───────────────────────────────────────
    os.makedirs('reports', exist_ok=True)
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['No Default','Default'])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False)
    plt.title(f'Best Model  |  AUC: {auc:.4f}')
    plt.tight_layout()
    plt.savefig('reports/confusion_matrix.png', dpi=100)
    plt.close()
    mlflow.log_artifact('reports/confusion_matrix.png')

    # ── 7. Feature importance plot ────────────────────────────────
    fi = pd.Series(best_model.feature_importances_,
                   index=X_train.columns).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    fi.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title('Feature Importance — Best Model')
    plt.tight_layout()
    plt.savefig('reports/feature_importance.png', dpi=100)
    plt.close()
    mlflow.log_artifact('reports/feature_importance.png')

    # ── 8. Save metrics JSON for CI summary ───────────────────────
    metrics = {
        "auc": round(auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }
    with open('reports/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    mlflow.log_artifact('reports/metrics.json')

    # ── 9. Register model in MLflow Model Registry ────────────────
    signature = infer_signature(X_train, y_pred)
    mlflow.xgboost.log_model(
        best_model, "model",
        signature=signature,
        registered_model_name=MODEL_NAME   # goes into registry
    )

    # ── 10. Save locally too ──────────────────────────────────────
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/model.pkl')
    joblib.dump(X_train.columns.tolist(), 'models/feature_columns.pkl')

    print(f"\n── Final Metrics ──────────────────────")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v}")
    print(f"\n  Registered in MLflow as: '{MODEL_NAME}'")
    print(f"  Saved locally: models/model.pkl")