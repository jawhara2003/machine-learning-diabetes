# src/modeling.py

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def train_final_model(X_train, y_train):
    param_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "solver": ["liblinear", "lbfgs"]
    }

    grid = GridSearchCV(
        LogisticRegression(max_iter=1000),
        param_grid,
        cv=5,
        scoring="accuracy"
    )

    grid.fit(X_train, y_train)

    return grid.best_estimator_
