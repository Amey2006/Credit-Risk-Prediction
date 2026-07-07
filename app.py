"""
Gradio app for the Credit Worthiness model.
Deployable as-is to Hugging Face Spaces (SDK: gradio).

Loads the trained Random Forest model + label encoders and exposes
a simple form UI where anyone can enter applicant details and get
a Good/Bad credit prediction with probability.
"""

import gradio as gr
import joblib
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Load trained artifacts (produced by credit_scoring_pipeline.py)
# ---------------------------------------------------------------------
model = joblib.load("models/best_model_random_forest.joblib")
encoders = joblib.load("models/label_encoders.joblib")

# Raw feature order the model expects, BEFORE the derived features are appended.
# This must mirror engineer_features()/encode_features() in the training script.
CATEGORICAL_COLS = list(encoders.keys())  # includes 'age_group' (derived) too

MODEL_FEATURES = list(model.feature_names_in_)


def engineer_and_encode(raw: dict) -> pd.DataFrame:
    df = pd.DataFrame([raw])

    # --- derived features (must match training pipeline) ---
    df["monthly_burden"] = df["Camt"] / df["Cdur"].replace(0, 1)
    df["age_group"] = pd.cut(
        df["age"], bins=[0, 25, 35, 45, 60, 100],
        labels=["<=25", "26-35", "36-45", "46-60", "60+"]
    ).astype(str)
    df["amt_per_dependent"] = df["Camt"] / df["Ndepend"].replace(0, 1)
    # medians hardcoded from training data (Cdur, Camt) so a single row can be scored
    df["high_risk_combo"] = ((df["Cdur"] > 18) & (df["Camt"] > 2320)).astype(int)

    # --- label-encode categoricals using the SAME encoders fit during training ---
    for col in CATEGORICAL_COLS:
        le = encoders[col]
        df[col] = le.transform(df[col].astype(str))

    # Reorder to match model's expected column order
    df = df[MODEL_FEATURES]
    return df


def predict(cbal, cdur, chist, cpur, camt, sbal, edur, inrate, msg, oparties,
            rdur, prop, age, inplans, htype, numcred, jobtype, ndepend,
            telephone, foreign):

    raw = {
        "Cbal": cbal, "Cdur": cdur, "Chist": chist, "Cpur": cpur, "Camt": camt,
        "Sbal": sbal, "Edur": edur, "InRate": inrate, "MSG": msg,
        "Oparties": oparties, "Rdur": rdur, "Prop": prop, "age": age,
        "inPlans": inplans, "Htype": htype, "NumCred": numcred,
        "JobType": jobtype, "Ndepend": ndepend, "telephone": telephone,
        "foreign": foreign,
    }

    X = engineer_and_encode(raw)
    proba_good = model.predict_proba(X)[0][1]
    label = "✅ Good credit risk" if proba_good >= 0.5 else "⚠️ Bad credit risk"

    return {
        "Good": float(proba_good),
        "Bad": float(1 - proba_good),
    }, f"{label}  (P(good) = {proba_good:.1%})"


def enc_choices(col):
    return list(encoders[col].classes_)


with gr.Blocks(title="Credit Worthiness Predictor") as demo:
    gr.Markdown(
        "# 💳 Credit Worthiness Predictor\n"
        "Enter applicant details to predict whether they're a **good** or "
        "**bad** credit risk. Model: Random Forest trained on the German "
        "Credit dataset (ROC-AUC ≈ 0.84)."
    )

    with gr.Row():
        with gr.Column():
            cbal = gr.Dropdown(enc_choices("Cbal"), label="Checking account balance", value=enc_choices("Cbal")[0])
            cdur = gr.Slider(4, 72, value=24, step=1, label="Credit duration (months)")
            chist = gr.Dropdown(enc_choices("Chist"), label="Credit history", value=enc_choices("Chist")[0])
            cpur = gr.Dropdown(enc_choices("Cpur"), label="Purpose of credit", value=enc_choices("Cpur")[0])
            camt = gr.Number(value=2500, label="Credit amount (Rs.)")
            sbal = gr.Dropdown(enc_choices("Sbal"), label="Savings balance", value=enc_choices("Sbal")[0])
            edur = gr.Dropdown(enc_choices("Edur"), label="Employment duration", value=enc_choices("Edur")[0])
            inrate = gr.Slider(1, 4, value=2, step=1, label="Installment rate (% of income)")
            msg = gr.Dropdown(enc_choices("MSG"), label="Marital status / sex", value=enc_choices("MSG")[0])
            oparties = gr.Dropdown(enc_choices("Oparties"), label="Other debtors/guarantors", value=enc_choices("Oparties")[0])

        with gr.Column():
            rdur = gr.Dropdown(enc_choices("Rdur"), label="Residence duration", value=enc_choices("Rdur")[0])
            prop = gr.Dropdown(enc_choices("Prop"), label="Property", value=enc_choices("Prop")[0])
            age = gr.Slider(18, 80, value=30, step=1, label="Age")
            inplans = gr.Dropdown(enc_choices("inPlans"), label="Other installment plans", value=enc_choices("inPlans")[0])
            htype = gr.Dropdown(enc_choices("Htype"), label="Housing", value=enc_choices("Htype")[0])
            numcred = gr.Slider(1, 4, value=1, step=1, label="Number of existing credits")
            jobtype = gr.Dropdown(enc_choices("JobType"), label="Job type", value=enc_choices("JobType")[0])
            ndepend = gr.Slider(1, 2, value=1, step=1, label="Number of dependents")
            telephone = gr.Dropdown(enc_choices("telephone"), label="Has telephone", value=enc_choices("telephone")[0])
            foreign = gr.Dropdown(enc_choices("foreign"), label="Foreign worker", value=enc_choices("foreign")[0])

    btn = gr.Button("Predict", variant="primary")
    label_out = gr.Label(label="Probability")
    text_out = gr.Textbox(label="Verdict")

    btn.click(
        predict,
        inputs=[cbal, cdur, chist, cpur, camt, sbal, edur, inrate, msg, oparties,
                rdur, prop, age, inplans, htype, numcred, jobtype, ndepend,
                telephone, foreign],
        outputs=[label_out, text_out],
    )

    gr.Markdown(
        "⚠️ *Educational demo only — trained on a small, dated benchmark "
        "dataset. Not suitable for real lending decisions.*"
    )

if __name__ == "__main__":
    demo.launch()
