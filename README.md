# Industrial Machine Failure Prediction

A machine learning project for predicting industrial machine failures using operating conditions such as temperature, rotational speed, torque, tool wear, and machine type.

## Problem Statement

Industrial machines can fail because of abnormal operating conditions and equipment wear. Unexpected failures can cause downtime, maintenance costs, and production losses.

This project develops a machine learning classification system that predicts whether a machine is likely to experience failure based on its observed operating parameters.

## Objective

To develop and evaluate machine learning models for industrial machine failure prediction and provide an interactive Streamlit application for real-time prediction.

## Dataset

The project uses the **AI4I 2020 Predictive Maintenance Dataset** from the UCI Machine Learning Repository.

* Dataset: AI4I 2020 Predictive Maintenance Dataset
* Instances: 10,000
* Columns: 14
* Target: `Machine failure`
* Dataset type: Synthetic predictive-maintenance data
* Missing values: None
* Task: Binary classification

The dataset contains machine operating parameters including:

* Product Type
* Air Temperature
* Process Temperature
* Rotational Speed
* Torque
* Tool Wear
* Machine Failure

Dataset source: UCI Machine Learning Repository
DOI: 10.24432/C5HS5C
License: CC BY 4.0

## Machine Learning Workflow

The project follows this workflow:

1. Dataset collection
2. Data preprocessing
3. Exploratory Data Analysis
4. Feature selection
5. Train-test split
6. Feature scaling and categorical encoding
7. Model training
8. Model evaluation
9. Best model selection
10. Prediction
11. Streamlit deployment

## Exploratory Data Analysis

The EDA includes:

* Dataset shape and structure
* Data types
* Missing-value analysis
* Duplicate-value analysis
* Target distribution
* Numerical feature distributions
* Boxplots
* Correlation analysis
* Failure rate by machine type
* Failure-mode analysis
* Outlier analysis

The EDA notebook is available at:

`notebooks/EDA.ipynb`

## Features Used

The final model uses the following features:

| Feature                 | Description              |
| ----------------------- | ------------------------ |
| Type                    | Machine/product type     |
| Air temperature [K]     | Air temperature          |
| Process temperature [K] | Process temperature      |
| Rotational speed [rpm]  | Machine rotational speed |
| Torque [Nm]             | Machine torque           |
| Tool wear [min]         | Tool usage/wear time     |

Target variable:

`Machine failure`

## Models Evaluated

Three classification algorithms were trained and compared:

1. Logistic Regression
2. Decision Tree
3. Random Forest

Class balancing was used during model training because machine failures are much less frequent than normal operating records.

## Model Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix

### Results

| Model               |   Accuracy |  Precision | Recall |   F1-Score |    ROC-AUC |
| ------------------- | ---------: | ---------: | -----: | ---------: | ---------: |
| Logistic Regression |     82.45% |     14.18% | 82.35% |     24.19% |     90.70% |
| Decision Tree       |     95.35% |     40.88% | 82.35% |     54.63% |     90.06% |
| Random Forest       | **96.85%** | **52.58%** | 75.00% | **61.82%** | **96.39%** |

The Random Forest model achieved the highest F1-score among the evaluated models and was saved as the final model.

## Class Distribution

The dataset contains:

* Normal operation: 9,661 records
* Machine failure: 339 records

This represents approximately:

* Normal: 96.61%
* Failure: 3.39%

Because the target classes are imbalanced, precision, recall, F1-score, and ROC-AUC are considered alongside accuracy.

## Project Structure

```text
Machine-Failure-Prediction/
│
├── data/
│   ├── ai4i2020.csv
│   ├── model_results.csv
│   └── eda/
│
├── models/
│   └── machine_failure_model.pkl
│
├── notebooks/
│   └── EDA.ipynb
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Model Training

The training pipeline:

* Loads the dataset
* Selects relevant features
* Splits the data into training and
