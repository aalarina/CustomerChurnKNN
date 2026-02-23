# ----- Imports -----
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score

# ----- Functions -----
def load_data(path="data/churn.csv"):
    """Load CSV dataset."""
    return pd.read_csv(path)

def preprocess_data(df):
    """Encode categorical features and drop non-informative columns."""
    df['international plan'] = df['international plan'].map({'no': 0, 'yes': 1})
    df['voice mail plan'] = df['voice mail plan'].map({'no': 0, 'yes': 1})
    df = df.drop(['phone number', 'state'], axis=1)
    df['churn'] = df['churn'].astype(int)
    return df

def split_data(df, target="churn", test_size=0.25, random_state=4):
    """Split dataset into training and test sets."""
    X = df.drop(target, axis=1)
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_baseline_model(X_train, y_train, X_test, y_test):
    """Train a baseline kNN model and evaluate its balanced accuracy."""
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = balanced_accuracy_score(y_test, y_pred)
    return model, score

def tune_hyperparameter_k(X_train, y_train):
    """Tune the number of neighbors (k) using GridSearchCV."""
    param_grid = {'n_neighbors': np.arange(1, 50, 2)}
    grid = GridSearchCV(
        KNeighborsClassifier(),
        param_grid,
        cv=5,
        scoring='balanced_accuracy'
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, grid.best_score_

def tune_hyperparameter_k_p(X_train, y_train):
    """Tune both k and Minkowski distance parameter p with weights='distance'."""
    param_grid = {
        'n_neighbors': np.arange(1, 10, 2),
        'p': np.linspace(1, 10, 10),
        'weights': ['distance']
    }
    grid = GridSearchCV(
        KNeighborsClassifier(metric='minkowski'),
        param_grid,
        cv=5,
        scoring='balanced_accuracy'
    )
    grid.fit(X_train, y_train)
    return grid.best_params_, grid.best_score_


# ----- Main Execution -----
def main():
    # Load and preprocess data
    df = load_data()
    df = preprocess_data(df)

    # Split data
    X_train, X_test, y_train, y_test = split_data(df)

    # Baseline model
    baseline_model, baseline_score = train_baseline_model(X_train, y_train, X_test, y_test)
    print("Baseline Balanced Accuracy:", baseline_score)

    # Hyperparameter tuning (k)
    best_k_model, best_k_params, best_k_score = tune_hyperparameter_k(X_train, y_train)
    print("Best k:", best_k_params)
    print("Best CV Score (k tuning):", best_k_score)

    # Evaluate best k model
    y_pred = best_k_model.predict(X_test)
    final_score = balanced_accuracy_score(y_test, y_pred)
    print("Final Balanced Accuracy (best k):", final_score)

    # Hyperparameter tuning (k + p)
    best_params_k_p, best_score_k_p = tune_hyperparameter_k_p(X_train, y_train)
    print("Best params (k + p):", best_params_k_p)
    print("Best CV Score (k + p tuning):", best_score_k_p)


if __name__ == "__main__":
    main()
