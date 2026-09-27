# Industrial Machine Failure Prediction

A machine learning project that predicts the likelihood of industrial machine failure using the AI4I 2020 Predictive Maintenance Dataset.

## Live Demo

https://sandanapm-machine-failure-prediction-app-ffsjwy.streamlit.app/

## GitHub Repository

https://github.com/SandanaPM/Machine-Failure-Prediction

## Project Objective

Develop an end-to-end machine learning solution to predict whether an industrial machine is likely to experience failure based on operating and maintenance parameters.

## Dataset

The project uses the AI4I 2020 Predictive Maintenance Dataset from the UCI Machine Learning Repository.

- Records: 10,000
- Features used:
  - Type
  - Air temperature [K]
  - Process temperature [K]
  - Rotational speed [rpm]
  - Torque [Nm]
  - Tool wear [min]
- Target: Machine failure
- Missing values: 0
- Duplicate records: 0

Dataset source:

https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

## Exploratory Data Analysis

The project investigates:

- Dataset structure and data types
- Missing values and duplicate records
- Class distribution
- Numerical feature distributions
- Outliers
- Feature relationships
- Correlation
- Failure rates
- Failure-mode indicators

EDA notebook:

`notebooks/EDA.ipynb`

## Machine Learning Models

Three classification models were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest

An 80/20 stratified train-test split was used.

Preprocessing includes:

- Standardization of numerical features
- One-hot encoding of the machine Type feature

The preprocessing and model are saved together as a pipeline.

## Model Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 82.45% | 14.18% | 82.35% | 24.19% | 90.70% |
| Decision Tree | 95.35% | 40.88% | 82.35% | 54.63% | 90.06% |
| Random Forest | 96.85% | 52.58% | 75.00% | 61.82% | 96.39% |

The Random Forest model was selected as the final prototype model based on the evaluation results.

## Streamlit Application

The Streamlit application allows users to enter machine operating parameters and receive:

- Predicted machine failure class
- Failure probability
- Machine status

The application uses the same preprocessing pipeline used during model training.

## Project Structure

    Machine-Failure-Prediction/
    ├── data/
    │   ├── ai4i2020.csv
    │   ├── model_results.csv
    │   └── eda/
    ├── models/
    │   └── machine_failure_model.pkl
    ├── notebooks/
    │   └── EDA.ipynb
    ├── src/
    │   ├── train.py
    │   └── predict.py
    ├── app.py
    ├── requirements.txt
    ├── README.md
    └── .gitignore

## Installation

Clone the repository:

    git clone https://github.com/SandanaPM/Machine-Failure-Prediction.git

Move into the project directory:

    cd Machine-Failure-Prediction

Install dependencies:

    pip install -r requirements.txt

## Run the Streamlit Application

    streamlit run app.py

## Run Model Training

    python src/train.py

## Run Prediction Test

    python src/predict.py

## Limitations

- The AI4I 2020 dataset is synthetic.
- The project is an educational prototype.
- The model is not intended to be used as a production industrial safety system.
- Real industrial deployment would require validation using real operational data.

## Future Scope

- Validate the model using real industrial sensor data.
- Include additional sensor and maintenance indicators.
- Investigate time-series failure patterns.
- Compare additional machine learning approaches.
- Further improve model evaluation and tuning.

## References

UCI Machine Learning Repository. AI4I 2020 Predictive Maintenance Dataset.

https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset