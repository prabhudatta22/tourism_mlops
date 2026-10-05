"""Model training: tune XGBoost, track experiments with MLflow, save the best model."""
import json
import os
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

BASE = Path(__file__).parent
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", BASE / "data" / "processed"))
MODEL_DIR = BASE / "model"

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
mlflow.set_experiment("tourism-package-prediction")

# ---------- 1. Load train/test splits from the workflow artifact
Xtrain = pd.read_csv(ARTIFACT_DIR / "Xtrain.csv")
Xtest = pd.read_csv(ARTIFACT_DIR / "Xtest.csv")
ytrain = pd.read_csv(ARTIFACT_DIR / "ytrain.csv").squeeze("columns")
ytest = pd.read_csv(ARTIFACT_DIR / "ytest.csv").squeeze("columns")
print(f"Loaded splits from {ARTIFACT_DIR}: train={Xtrain.shape}, test={Xtest.shape}")

# ---------- 2. Define model
cat_cols = Xtrain.select_dtypes(exclude="number").columns.tolist()
num_cols = Xtrain.select_dtypes(include="number").columns.tolist()

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols),
])

pos_weight = (ytrain == 0).sum() / (ytrain == 1).sum()   # counter the ~19% positive class
model = Pipeline([
    ("prep", preprocessor),
    ("xgb", XGBClassifier(scale_pos_weight=pos_weight, eval_metric="logloss",
                          random_state=42, n_jobs=-1)),
])

# ---------- 3. Define tuning parameters
param_grid = {
    "xgb__n_estimators": [100, 200],
    "xgb__max_depth": [4, 6],
    "xgb__learning_rate": [0.05, 0.1],
    "xgb__subsample": [0.8, 1.0],
    "xgb__colsample_bytree": [0.7, 1.0],
}

with mlflow.start_run(run_name="xgb-gridsearch"):
    # ---------- 4. Tune
    search = GridSearchCV(model, param_grid, scoring="f1", n_jobs=-1,
                          cv=StratifiedKFold(5, shuffle=True, random_state=42))
    search.fit(Xtrain, ytrain)

    # ---------- 5. Log ALL tuned parameter combinations as nested runs
    res = search.cv_results_
    for i, params in enumerate(res["params"]):
        with mlflow.start_run(run_name=f"candidate-{i:02d}", nested=True):
            mlflow.log_params({k.replace("xgb__", ""): v for k, v in params.items()})
            mlflow.log_metric("cv_f1_mean", res["mean_test_score"][i])
            mlflow.log_metric("cv_f1_std", res["std_test_score"][i])
    print(f"Logged {len(res['params'])} candidate parameter sets to MLflow")

    best = search.best_estimator_
    mlflow.log_params({f"best_{k.replace('xgb__', '')}": v for k, v in search.best_params_.items()})
    mlflow.log_param("scale_pos_weight", round(float(pos_weight), 3))
    mlflow.log_metric("best_cv_f1", search.best_score_)

    # ---------- 6. Evaluate
    metrics = {}
    for split, X, y in [("train", Xtrain, ytrain), ("test", Xtest, ytest)]:
        pred, proba = best.predict(X), best.predict_proba(X)[:, 1]
        metrics.update({
            f"{split}_accuracy": accuracy_score(y, pred),
            f"{split}_precision": precision_score(y, pred),
            f"{split}_recall": recall_score(y, pred),
            f"{split}_f1": f1_score(y, pred),
            f"{split}_roc_auc": roc_auc_score(y, proba),
        })
    mlflow.log_metrics(metrics)

    print("Best params :", search.best_params_)
    print(f"Best CV F1  : {search.best_score_:.4f}")
    print(pd.DataFrame({s: {m: metrics[f'{s}_{m}'] for m in
                            ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']}
                        for s in ['train', 'test']}).round(4).to_string())
    print("\nTest classification report:\n", classification_report(ytest, best.predict(Xtest)))

    # ---------- 7. Save best model (committed to the repo by the workflow)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best, MODEL_DIR / "best_model.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(
        {"trained_by_run": os.getenv("GITHUB_RUN_NUMBER", "local"), "best_params": search.best_params_, **{k: round(v, 4) for k, v in metrics.items()}}, indent=2))
    mlflow.log_artifact(str(MODEL_DIR / "best_model.joblib"), artifact_path="model")
    print("Saved best model to", MODEL_DIR / "best_model.joblib")
