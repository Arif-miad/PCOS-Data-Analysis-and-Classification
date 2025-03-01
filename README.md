# PCOS Data Analysis and Classification

## Overview
This project focuses on analyzing and classifying Polycystic Ovary Syndrome (PCOS) using a dataset containing various health-related parameters. The primary goal is to perform data analysis, visualization, and classification using different machine learning models.

## Dataset Description
The dataset consists of the following features:
- **Age**: Age of the individual.
- **BMI**: Body Mass Index.
- **Menstrual Irregularity**: Binary indicator (1 for irregular, 0 for regular).
- **Testosterone Level (ng/dL)**: Testosterone concentration in the blood.
- **Antral Follicle Count**: Number of antral follicles observed.
- **PCOS Diagnosis**: Target variable (1 for PCOS, 0 for non-PCOS).

## Project Workflow
1. **Data Preprocessing**:
   - Handling missing values (if any).
   - Standardizing numerical features.
   - Splitting the dataset into training and test sets.
2. **Data Visualization**:
   - Distribution plots for each feature.
   - Correlation heatmaps.
   - Boxplots for outlier detection.
   - Scatter plots and pair plots.
3. **Machine Learning Models**:
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Support Vector Machine (SVM)
   - Naive Bayes
   - Decision Tree
   - Random Forest
   - Gradient Boosting
   - AdaBoost
   - Extra Trees Classifier
   - XGBoost
4. **Model Evaluation**:
   - Classification report (Precision, Recall, F1-score, Accuracy)
   - AUC-ROC Curve for model comparison.
   - Feature importance analysis (for tree-based models).

## Implementation
The code follows these key steps:

```python
# Data Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model Training and Evaluation
for name, model in top_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    print(f"{name} Classification Report:\n", classification_report(y_test, y_pred))
```

## Results
- The models were compared based on AUC scores, and the ROC curves were plotted.
- The best-performing models were identified based on their classification metrics.

## Conclusion
This project provides insights into PCOS diagnosis using machine learning. The results can help in understanding the most significant features affecting PCOS prediction.

## Installation & Dependencies
To run this project, install the required libraries:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn xgboost
```

## Author
[Arif Miah]

## Connect with Me  
- Kaggle: [Your Kaggle Profile](https://www.kaggle.com/miadul)
- LinkedIn: [Your LinkedIn Profile](www.linkedin.com/in/arif-miah-8751bb217)  
 


