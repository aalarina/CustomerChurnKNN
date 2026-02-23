# ----- Imports -----
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score


# ----- Load Data -----
df = pd.read_csv("data/churn.csv")


# ----- Preprocessing -----
df['international plan'] = df['international plan'].map({'no': 0, 'yes': 1})
df['voice mail plan'] = df['voice mail plan'].map({'no': 0, 'yes': 1})

df = df.drop(['phone number', 'state'], axis=1)
df['churn'] = df['churn'].astype(int)


# ----- Split Data -----
X = df.drop('churn', axis=1)
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=4
)


# ----- Baseline Model -----
model = KNeighborsClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
baseline_score = balanced_accuracy_score(y_test, y_pred)

print("Baseline Balanced Accuracy:", baseline_score)


# ----- Hyperparameter Tuning (k) -----
param_grid = {
    'n_neighbors': np.arange(1, 50, 2)
}

grid = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=5,
    scoring='balanced_accuracy'
)

grid.fit(X_train, y_train)

best_k_model = grid.best_estimator_

print("Best k:", grid.best_params_)
print("CV Score:", grid.best_score_)


# ----- Evaluate Best Model -----
y_pred = best_k_model.predict(X_test)
final_score = balanced_accuracy_score(y_test, y_pred)

print("Final Balanced Accuracy:", final_score)


# ----- Tuning distance metric (p) -----
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

print("Best params (k + p):", grid.best_params_)
print("Best CV score:", grid.best_score_)
