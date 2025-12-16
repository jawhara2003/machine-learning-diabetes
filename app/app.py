import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# CONFIGURATION
# =========================
st.set_page_config(
    page_title="Prédiction du Diabète",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Application de Prédiction du Diabète")

st.write(
    "Importer un fichier CSV contenant les données des patients "
    "afin de prédire s’ils sont diabétiques ou non."
)

# =========================
# CHARGER LE MODÈLE
# =========================
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/diabetes_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()


# =========================
# COLONNES ATTENDUES
# =========================
EXPECTED_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

# =========================
# PRÉTRAITEMENT
# =========================
def preprocess_data(df):
    cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)
    df.fillna(df.median(), inplace=True)
    return df

# =========================
# IMPORT CSV
# =========================
uploaded_file = st.file_uploader("📂 Importer un fichier CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Aperçu des données")
    st.dataframe(df)

    if list(df.columns) != EXPECTED_COLUMNS:
        st.error("❌ Format du fichier incorrect")
        st.write("Colonnes attendues :", EXPECTED_COLUMNS)
    else:
        st.success("✅ Fichier valide")

        if st.button("🔍 Lancer la prédiction"):
            df_processed = preprocess_data(df)
            df_scaled = scaler.transform(df_processed)
            predictions = model.predict(df_scaled)

            df_results = df.copy()
            df_results["Résultat"] = [
                "Diabétique" if p == 1 else "Non diabétique"
                for p in predictions
            ]

            st.subheader("📊 Résultats")
            st.dataframe(df_results)

            st.success("✅ Prédiction terminée")
