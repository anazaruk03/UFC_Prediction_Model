import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ---------------------------------------------------------
# UFC Fight Outcome Prediction – Streamlit App
# Loads a pre-trained model (ufc_fight_predictor_tuned.pkl)
# and predicts P(Fighter A Wins) for uploaded fights.
# ---------------------------------------------------------

MODEL_PATH = Path("ufc_fight_predictor_tuned.pkl")
TARGET = "fighter_a_won"

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' not found. "
            "Make sure it is in the same folder as app.py."
        )
    model = joblib.load(MODEL_PATH)
    return model

def preprocess_input(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Mirror the training-time preprocessing at the *column* level:
    - Drop ID / metadata columns that were removed before modeling
    - Drop the target column if it's present
    The heavy lifting (scaling, one-hot encoding, etc.) is handled
    inside the saved sklearn Pipeline's 'prep' step.
    """
    cols_to_drop = [
        "fight_key",
        "fight_id",
        "event_id",
        "event_date",
        "event_name",
        "promotion",
        "fighter_a_id",
        "fighter_a_name",
        "fighter_b_id",
        "fighter_b_name",
        "title_fight",
        "fight_end_round",
        "fight_result_type",
        "fight_duration",
    ]
    # Only drop columns that actually exist in the uploaded data
    cols_to_drop = [c for c in cols_to_drop if c in df_raw.columns]
    df_model = df_raw.drop(columns=cols_to_drop, errors="ignore").copy()

    # If the target column is present (e.g., historical labeled data),
    # drop it for prediction time.
    if TARGET in df_model.columns:
        df_model = df_model.drop(columns=[TARGET])

    return df_model

def main():
    st.title("🥊 UFC Fight Outcome Predictor")
    st.write(
        """
        This app loads your tuned machine learning model and predicts  
        **P(Fighter A Wins)** for each fight in an uploaded CSV.

        **Instructions**
        1. Place `app.py` and `ufc_fight_predictor_tuned.pkl` in the same folder.
        2. Deploy this repo to Streamlit Cloud.
        3. Upload a CSV with the same structure as your modeling dataset  
           (it can include the target column `fighter_a_won` or not).
        4. The app will output a probability for each row and let you download the results.
        """
    )

    st.sidebar.header("Settings")
    show_preview = st.sidebar.checkbox("Show data preview", value=True)
    threshold = st.sidebar.slider(
        "Decision threshold for Fighter A win (for labeling only)",
        0.0,
        1.0,
        0.5,
        0.01,
    )

    uploaded_file = st.file_uploader(
        "Upload fight data as CSV",
        type=["csv"],
        help="Use the same columns as in your training data.",
    )

    if uploaded_file is None:
        st.info("⬆️ Upload a CSV file to get started.")
        return

    # Read uploaded CSV
    try:
        df_raw = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return

    st.success(f"Loaded data with shape: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")

    if show_preview:
        st.subheader("Data Preview (first 10 rows)")
        st.dataframe(df_raw.head(10))

    # Preprocess to match model expectations
    df_features = preprocess_input(df_raw)

    if df_features.empty:
        st.error("After preprocessing, there are no feature columns left. Check your input file.")
        return

    st.write(
        f"Using **{df_features.shape[1]}** feature columns for prediction "
        f"({df_features.shape[0]} rows)."
    )

    if st.button("Run Predictions"):
        try:
            model = load_model()
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return

        try:
            # Assumes the saved object is a sklearn Pipeline with predict_proba
            probs = model.predict_proba(df_features)[:, 1]
        except Exception as e:
            st.error(
                "Error during prediction. Make sure the uploaded CSV has the same structure "
                "used when training the model.\n\n"
                f"Details: {e}"
            )
            return

        df_results = df_raw.copy()
        df_results["P_FighterA_Win"] = probs
        df_results["Predicted_Label"] = (df_results["P_FighterA_Win"] >= threshold).astype(int)

        st.subheader("Prediction Results (first 20 rows)")
        st.dataframe(df_results.head(20))

        st.subheader("Summary")
        st.write(
            f"Average predicted P(Fighter A Wins): **{df_results['P_FighterA_Win'].mean():.3f}**"
        )
        win_rate = df_results["Predicted_Label"].mean()
        st.write(
            f"Fraction of fights where Fighter A is predicted to win (threshold={threshold:.2f}): "
            f"**{win_rate:.3f}**"
        )

        # Download predictions as CSV
        csv_bytes = df_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download predictions as CSV",
            data=csv_bytes,
            file_name="ufc_fight_predictions_with_probs.csv",
            mime="text/csv",
        )

if __name__ == "__main__":
    main()
