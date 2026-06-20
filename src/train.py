from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


def find_dataset(data_dir: Path) -> Path:
    candidates = [
        data_dir / "HR-Employee-Attrition.csv",
        data_dir / "WA_Fn-UseC_-HR-Employee-Attrition.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "CSV not found. Put the dataset at data/HR-Employee-Attrition.csv "
        "(or data/WA_Fn-UseC_-HR-Employee-Attrition.csv)."
    )


def tune_threshold(y_true: np.ndarray, proba: np.ndarray, target_recall: float) -> tuple[float, dict]:
    thresholds = np.linspace(0.05, 0.95, 91)
    rows: list[dict] = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, pred, average="binary", zero_division=0
        )
        rows.append({"threshold": float(t), "precision": float(precision), "recall": float(recall), "f1": float(f1)})

    thr_df = pd.DataFrame(rows)
    candidates = thr_df[thr_df["recall"] >= target_recall].copy()
    if len(candidates) == 0:
        chosen = thr_df.sort_values(["recall", "f1"], ascending=False).iloc[0]
        note = "No threshold met target recall; chose best recall instead."
    else:
        chosen = candidates.sort_values(["f1", "precision"], ascending=False).iloc[0]
        note = "Met target recall; chose best F1 under the recall constraint."

    chosen_threshold = float(chosen["threshold"])
    meta = {"note": note, "table": thr_df.to_dict(orient="records")}
    return chosen_threshold, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-recall", type=float, default=0.80, help="Recall target for threshold tuning.")
    args = parser.parse_args()

    sns.set_theme(style="whitegrid")

    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    models_dir = project_root / "models"
    fig_dir = project_root / "reports" / "figures"
    models_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    csv_path = find_dataset(data_dir)
    print("Using dataset:", csv_path)
    df = pd.read_csv(csv_path)

    # Target encode
    df_model = df.copy()
    df_model["Attrition"] = df_model["Attrition"].map({"Yes": 1, "No": 0}).astype(int)

    # Drop irrelevant columns
    drop_cols = ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]
    drop_cols = [c for c in drop_cols if c in df_model.columns]
    df_model = df_model.drop(columns=drop_cols)

    X = df_model.drop(columns=["Attrition"])
    y = df_model["Attrition"].astype(int)

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    # Train / test split (stratified)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    # Validation split for threshold tuning
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42
    )

    numeric_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[("num", numeric_pipe, num_cols), ("cat", categorical_pipe, cat_cols)],
        remainder="drop",
    )

    smote = SMOTE(random_state=42)

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"recall": "recall", "precision": "precision", "f1": "f1", "roc_auc": "roc_auc"}

    cv_rows: list[dict] = []
    pipelines: dict[str, ImbPipeline] = {}

    print("\nRunning CV (recall prioritized)...\n")
    for name, clf in models.items():
        pipe = ImbPipeline(steps=[("preprocess", preprocess), ("smote", smote), ("model", clf)])
        pipelines[name] = pipe
        scores = cross_validate(
            pipe,
            X_train_full,
            y_train_full,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )
        cv_rows.append(
            {
                "model": name,
                "recall_mean": float(np.mean(scores["test_recall"])),
                "precision_mean": float(np.mean(scores["test_precision"])),
                "f1_mean": float(np.mean(scores["test_f1"])),
                "roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
            }
        )

    cv_results = pd.DataFrame(cv_rows).sort_values(by="recall_mean", ascending=False)
    print(cv_results)
    cv_results.to_csv(models_dir / "cv_results.csv", index=False)

    best_name = str(cv_results.iloc[0]["model"])
    best_pipe = pipelines[best_name]
    print("\nBest by CV recall:", best_name)

    # Fit on train and tune threshold on valid
    best_pipe.fit(X_train, y_train)
    valid_proba = best_pipe.predict_proba(X_valid)[:, 1]
    chosen_threshold, threshold_meta = tune_threshold(y_valid.to_numpy(), valid_proba, args.target_recall)
    print("\nChosen threshold:", chosen_threshold)

    # Plot threshold curve (saved)
    thr_df = pd.DataFrame(threshold_meta["table"])
    plt.figure(figsize=(7, 4))
    plt.plot(thr_df["threshold"], thr_df["recall"], label="Recall")
    plt.plot(thr_df["threshold"], thr_df["precision"], label="Precision")
    plt.plot(thr_df["threshold"], thr_df["f1"], label="F1")
    plt.axhline(args.target_recall, color="red", linestyle="--", label=f"Target recall={args.target_recall}")
    plt.axvline(chosen_threshold, color="black", linestyle=":", label=f"Chosen t={chosen_threshold:.2f}")
    plt.xlabel("Threshold")
    plt.title("Threshold tuning on validation set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "threshold_tuning.png", dpi=200)
    plt.close()

    # Final fit on full train and evaluate on test
    best_pipe.fit(X_train_full, y_train_full)
    test_proba = best_pipe.predict_proba(X_test)[:, 1]
    y_pred = (test_proba >= chosen_threshold).astype(int)

    roc_auc = float(roc_auc_score(y_test, test_proba))
    report = classification_report(y_test, y_pred, digits=3, output_dict=True, zero_division=0)
    print("\nROC-AUC:", roc_auc)
    print("\nClassification report (threshold tuned):\n")
    print(classification_report(y_test, y_pred, digits=3, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(values_format="d")
    plt.title(f"Confusion Matrix ({best_name}, t={chosen_threshold:.2f})")
    plt.tight_layout()
    plt.savefig(fig_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    # SHAP (best-effort)
    shap_ok = False
    try:
        import shap  # noqa: WPS433

        pre = best_pipe.named_steps["preprocess"]
        model = best_pipe.named_steps["model"]

        X_test_proc = pre.transform(X_test)
        X_train_proc = pre.transform(X_train_full)
        feature_names = pre.get_feature_names_out()

        sample_n = min(200, X_test_proc.shape[0])
        X_shap = X_test_proc[:sample_n]

        if best_name in ["RandomForest", "XGBoost"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_shap)
            if isinstance(shap_values, list):
                shap_values_plot = shap_values[1]
            else:
                shap_values_plot = shap_values
        else:
            explainer = shap.LinearExplainer(model, X_train_proc, feature_names=feature_names)
            shap_values_plot = explainer.shap_values(X_shap)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_plot, X_shap, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(fig_dir / "shap_summary.png", dpi=250, bbox_inches="tight")
        plt.close()
        shap_ok = True
    except Exception as e:
        print("\nSHAP step skipped (error):", e)

    # Export artifacts for Streamlit
    joblib.dump(best_pipe, models_dir / "attrition_pipeline.pkl")

    (models_dir / "threshold.json").write_text(
        json.dumps(
            {
                "model": best_name,
                "threshold": chosen_threshold,
                "target_recall": args.target_recall,
                "note": threshold_meta["note"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    schema = {
        "columns": X.columns.tolist(),
        "categorical": {},
        "numeric": {},
        "defaults": {},
    }
    for c in cat_cols:
        cats = sorted([x for x in X[c].dropna().unique().tolist()])
        schema["categorical"][c] = cats
        schema["defaults"][c] = X[c].mode().iloc[0]

    for c in num_cols:
        schema["numeric"][c] = {
            "min": float(np.nanmin(X[c].values)),
            "max": float(np.nanmax(X[c].values)),
        }
        schema["defaults"][c] = float(np.nanmedian(X[c].values))

    (models_dir / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    metrics = {
        "best_model": best_name,
        "threshold": chosen_threshold,
        "roc_auc": roc_auc,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "shap_summary_saved": shap_ok,
    }
    (models_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nSaved:")
    print(" -", models_dir / "attrition_pipeline.pkl")
    print(" -", models_dir / "threshold.json")
    print(" -", models_dir / "schema.json")
    print(" -", models_dir / "metrics.json")
    print("Figures:")
    print(" -", fig_dir)


if __name__ == "__main__":
    main()

