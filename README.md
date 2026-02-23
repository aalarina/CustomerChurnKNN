# KNN Classification on Customer Churn Dataset

**Dataset:** [Churn in Telecoms Dataset](https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset)

## Overview
This project explores a supervised machine learning approach to predicting customer churn using the k-Nearest Neighbors (kNN) algorithm.

Customer churn prediction is a common classification task in industry, where the objective is to identify users likely to discontinue a service. The dataset contains a mix of numerical and categorical features describing customer behavior.

The project focuses on building a baseline model, improving it through hyperparameter tuning, and analyzing its performance.

The analysis includes:
- Preprocessing categorical and numerical features
- Train/test split
- Training kNN classifiers
- Hyperparameter tuning (number of neighbors, distance metric)
- Cross-validation and evaluation using balanced accuracy
- Visualizations of validation curves

## Repository Structure

```
CustomerChurnKNN/
├── CustomerChurnKNN.ipynb     # main analysis notebook
├── README.md
├── requirements.txt
└── LICENSE
```

## Methodology

**Data Preprocessing:** Converted categorical features into numerical representations and removed non-informative features (e.g., phone number, state).

**Model Training:** Implemented the KNeighborsClassifier from scikit-learn, trained an initial baseline model and evaluated performance on a validation split.

**Hyperparameter Tuning:** To improve model performance, number of neighbors (k) and distance metric (Minkowski distance parameter p) were optimized.
Grid search with cross-validation (GridSearchCV) was used to identify optimal values.

**Evaluation Metric:** Balanced Accuracy was used due to class imbalance in the dataset.
This metric provides a more reliable evaluation compared to standard accuracy.

## Results

- Best hyperparameters identified using cross-validation

- Performance evaluated on validation set

- Visualization of accuracy vs number of neighbors

- Observed that model performance remained below 0.7, indicating limitations of kNN for this dataset

## Conclusion

This project demonstrates:

- Practical application of the kNN algorithm

- Importance of preprocessing for distance-based models

- The role of hyperparameter tuning in improving performance

- Limitations of simpler models on complex datasets

## License

This project is licensed under the MIT License.
