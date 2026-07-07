"""
Credit Worthiness Prediction Pipeline
=====================================
Predicts whether an individual is a good or bad credit risk using
classical ML classifiers (Logistic Regression, Decision Tree, Random Forest).

Dataset: CreditWorthiness.xlsx (German Credit Data — 1000 rows, 20 features
+ 1 target 'creditScore' with classes 'good' / 'bad').

Pipeline stages:
    1. Load & explore data
    2. Feature engineering (encode categoricals, derive ratio features)
    3. Train/test split (stratified)
    4. Train Logistic Regression, Decision Tree, Random Forest
    5. Evaluate: Precision, Recall, F1, ROC-AUC, Confusion Matrix
    6. Save comparison plots + best model
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)

sns.set_style("whitegrid")
RANDOM_STATE = 42

DATA_PATH = "data/CreditWorthiness.xlsx"
OUT_DIR = "outputs"
MODEL_DIR = "models"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------
def load_data(path):
    df = pd.read_excel(path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------------
def engineer_features(df):
    df = df.copy()

    # Target encoding: good -> 1 (creditworthy), bad -> 0
    df["target"] = df["creditScore"].map({"good": 1, "bad": 0})
    df.drop(columns=["creditScore"], inplace=True)

    # --- Derived numeric features from financial history ---
    # Monthly loan burden = credit amount / duration (proxy for repayment pressure)
    df["monthly_burden"] = df["Camt"] / df["Cdur"].replace(0, 1)

    # Age buckets capture life-stage risk differences
    df["age_group"] = pd.cut(
        df["age"], bins=[0, 25, 35, 45, 60, 100],
        labels=["<=25", "26-35", "36-45", "46-60", "60+"]
    ).astype(str)

    # Credit amount per dependent (financial strain indicator)
    df["amt_per_dependent"] = df["Camt"] / df["Ndepend"].replace(0, 1)

    # Flag long-duration high-amount loans (higher risk combo)
    df["high_risk_combo"] = (
        (df["Cdur"] > df["Cdur"].median()) &
        (df["Camt"] > df["Camt"].median())
    ).astype(int)

    return df


def encode_features(df, categorical_cols):
    df = df.copy()
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


# ---------------------------------------------------------------------
# 3. TRAIN / EVALUATE
# ---------------------------------------------------------------------
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    cm = confusion_matrix(y_test, y_pred)
    return metrics, cm, y_proba, y_pred


def main():
    # ---- Load & feature engineer ----
    df_raw = load_data(DATA_PATH)
    df = engineer_features(df_raw)

    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    df_enc, encoders = encode_features(df, categorical_cols)

    X = df_enc.drop(columns=["target"])
    y = df_enc["target"]

    print(f"\nFeature count after engineering: {X.shape[1]}")
    print(f"Class balance -> good: {(y==1).sum()}, bad: {(y==0).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---- Define models ----
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=10,
            class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    results = []
    cms = {}
    roc_data = {}
    trained_models = {}

    for name, model in models.items():
        # Logistic Regression benefits from scaled features; trees do not need it
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            metrics, cm, y_proba, y_pred = evaluate_model(name, model, X_test_scaled, y_test)
        else:
            model.fit(X_train, y_train)
            metrics, cm, y_proba, y_pred = evaluate_model(name, model, X_test, y_test)

        results.append(metrics)
        cms[name] = cm
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = (fpr, tpr, metrics["roc_auc"])
        trained_models[name] = model

        print(f"\n{name}")
        print(classification_report(y_test, y_pred, target_names=["bad", "good"]))

    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    print("\n=== MODEL COMPARISON ===")
    print(results_df.to_string(index=False))
    results_df.to_csv(f"{OUT_DIR}/model_comparison.csv", index=False)

    # ---- Feature importance (Random Forest) ----
    rf = trained_models["Random Forest"]
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    importances.to_csv(f"{OUT_DIR}/feature_importance.csv", header=["importance"])

    # ================= PLOTS =================
    # 1. ROC curves
    plt.figure(figsize=(7, 6))
    for name, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/roc_curves.png", dpi=150)
    plt.close()

    # 2. Confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (name, cm) in zip(axes, cms.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["bad", "good"], yticklabels=["bad", "good"], ax=ax, cbar=False)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/confusion_matrices.png", dpi=150)
    plt.close()

    # 3. Metric comparison bar chart
    plot_df = results_df.set_index("model")[["precision", "recall", "f1_score", "roc_auc"]]
    plot_df.plot(kind="bar", figsize=(9, 5.5), rot=0)
    plt.title("Precision / Recall / F1 / ROC-AUC by Model")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/metric_comparison.png", dpi=150)
    plt.close()

    # 4. Top feature importances
    plt.figure(figsize=(8, 6))
    top_feats = importances.head(15)[::-1]
    plt.barh(top_feats.index, top_feats.values, color="#4C72B0")
    plt.title("Top 15 Feature Importances (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/feature_importance.png", dpi=150)
    plt.close()

    # ---- Save best model ----
    best_model_name = results_df.iloc[0]["model"]
    best_model = trained_models[best_model_name]
    joblib.dump(best_model, f"{MODEL_DIR}/best_model_{best_model_name.replace(' ', '_').lower()}.joblib")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.joblib")
    joblib.dump(encoders, f"{MODEL_DIR}/label_encoders.joblib")

    with open(f"{OUT_DIR}/summary.json", "w") as f:
        json.dump({
            "best_model": best_model_name,
            "results": results_df.to_dict(orient="records"),
            "top_features": importances.head(10).to_dict(),
        }, f, indent=2)

    print(f"\nBest model: {best_model_name}")
    print(f"All outputs saved to '{OUT_DIR}/' and model artifacts to '{MODEL_DIR}/'")


if __name__ == "__main__":
    main()
