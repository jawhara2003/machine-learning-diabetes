# src/evaluation.py

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    results = {}

    results["accuracy"] = accuracy_score(y_test, y_pred)
    results["confusion_matrix"] = confusion_matrix(y_test, y_pred)
    results["classification_report"] = classification_report(y_test, y_pred)

    TN, FP, FN, TP = results["confusion_matrix"].ravel()

    results["precision"] = TP / (TP + FP)
    results["recall"] = TP / (TP + FN)
    results["specificity"] = TN / (TN + FP)
    results["f1_score"] = 2 * (
        results["precision"] * results["recall"]
    ) / (results["precision"] + results["recall"])

    results["fpr"] = FP / (FP + TN)
    results["fnr"] = FN / (FN + TP)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        results["auc"] = roc_auc_score(y_test, y_prob)

    return results
