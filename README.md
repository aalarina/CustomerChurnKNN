# KNN Classification on Customer Churn Dataset

**Dataset:** [Link to dataset on Google Drive](https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset)

## Overview
This project demonstrates supervised learning using the k-Nearest Neighbors (kNN) algorithm. 
The goal is to predict customer churn (binary classification) using various features from the dataset.

The analysis includes:
- Preprocessing categorical and numerical features
- Train/test split
- Training kNN classifiers
- Hyperparameter tuning (number of neighbors, distance metric)
- Cross-validation and evaluation using balanced accuracy
- Visualizations of validation curves

## Methodology

### Data Preprocessing

- Encoded categorical variables into numeric format

- Removed non-informative columns (e.g., phone number, state)

### Model Training

- Used KNeighborsClassifier

- Evaluated initial model performance

- Hyperparameter Tuning

**Optimized:**

- Number of neighbors (k)

- Distance metric (p parameter)

Used GridSearchCV with cross-validation

### Evaluation Metric

Balanced Accuracy, suitable for imbalanced datasets
