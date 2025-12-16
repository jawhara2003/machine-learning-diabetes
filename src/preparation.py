# src/preparation.py

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def prepare_data(df):
    # 1. Sélection
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # 2. Nettoyage
    cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    X[cols_with_zero] = X[cols_with_zero].replace(0, np.nan)
    X.fillna(X.median(), inplace=True)

    # 3. Intégration
    df_clean = X.copy()
    df_clean["Outcome"] = y

    # 4. Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5. Formatage
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test, scaler, df_clean
