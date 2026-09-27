import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

DATA_PATH = "data/ai4i2020.csv"
MODEL_PATH = "models/machine_failure_model.pkl"

os.makedirs("models", exist_ok=True)
os.makedirs("data/eda", exist_ok=True)

print("=" * 60)
print("MACHINE FAILURE PREDICTION")
print("=" * 60)

# Load dataset
df = pd.read_csv(DATA_PATH)

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
for column in df.columns:
    print("-", column)

print("\nFirst 5 Rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nTarget Distribution:")
print(df["Machine failure"].value_counts())

print("\nTarget Percentage:")
print(df["Machine failure"].value_counts(normalize=True) * 100)

# Remove duplicate rows
df = df.drop_duplicates()

# Features selected for prediction
features = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

target = "Machine failure"

X = df[features]
y = df[target]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

categorical_features = ["Type"]

numeric_features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

# Models
models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=1000
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=8,
        random_state=42,
        class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    )
}

results = []

best_model = None
best_model_name = None
best_f1 = -1

print("\n" + "=" * 60)
print("MODEL TRAINING")
print("=" * 60)

for model_name, classifier in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )
    roc_auc = roc_auc_score(y_test, y_prob)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    })

    print("\n" + "-" * 60)
    print(model_name)
    print("-" * 60)

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC-AUC  :", round(roc_auc, 4))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Failure"],
        yticklabels=["Normal", "Failure"]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    plt.savefig(
        f"data/eda/{safe_name}_confusion_matrix.png",
        dpi=300
    )
    plt.close()

    if f1 > best_f1:
        best_f1 = f1
        best_model = pipeline
        best_model_name = model_name

# Results table
results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))

results_df.to_csv(
    "data/model_results.csv",
    index=False
)

# Save best model
joblib.dump(
    best_model,
    MODEL_PATH
)

print("\nBest Model:", best_model_name)
print("Best F1 Score:", round(best_f1, 4))
print("Model saved to:", MODEL_PATH)

# EDA plots

# Target distribution
plt.figure(figsize=(6, 4))
sns.countplot(
    data=df,
    x="Machine failure"
)
plt.title("Machine Failure Distribution")
plt.xlabel("Machine Failure")
plt.ylabel("Number of Machines")
plt.tight_layout()
plt.savefig(
    "data/eda/machine_failure_distribution.png",
    dpi=300
)
plt.close()

# Air temperature
plt.figure(figsize=(7, 5))
sns.boxplot(
    data=df,
    x="Machine failure",
    y="Air temperature [K]"
)
plt.title("Air Temperature vs Machine Failure")
plt.tight_layout()
plt.savefig(
    "data/eda/air_temperature_failure.png",
    dpi=300
)
plt.close()

# Process temperature
plt.figure(figsize=(7, 5))
sns.boxplot(
    data=df,
    x="Machine failure",
    y="Process temperature [K]"
)
plt.title("Process Temperature vs Machine Failure")
plt.tight_layout()
plt.savefig(
    "data/eda/process_temperature_failure.png",
    dpi=300
)
plt.close()

# Rotational speed
plt.figure(figsize=(7, 5))
sns.boxplot(
    data=df,
    x="Machine failure",
    y="Rotational speed [rpm]"
)
plt.title("Rotational Speed vs Machine Failure")
plt.tight_layout()
plt.savefig(
    "data/eda/rotational_speed_failure.png",
    dpi=300
)
plt.close()

# Torque
plt.figure(figsize=(7, 5))
sns.boxplot(
    data=df,
    x="Machine failure",
    y="Torque [Nm]"
)
plt.title("Torque vs Machine Failure")
plt.tight_layout()
plt.savefig(
    "data/eda/torque_failure.png",
    dpi=300
)
plt.close()

# Tool wear
plt.figure(figsize=(7, 5))
sns.boxplot(
    data=df,
    x="Machine failure",
    y="Tool wear [min]"
)
plt.title("Tool Wear vs Machine Failure")
plt.tight_layout()
plt.savefig(
    "data/eda/tool_wear_failure.png",
    dpi=300
)
plt.close()

# Correlation heatmap
numeric_df = df.select_dtypes(include=["number"])

plt.figure(figsize=(10, 7))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(
    "data/eda/correlation_heatmap.png",
    dpi=300
)
plt.close()

print("\nEDA graphs saved in: data/eda/")
print("Model results saved in: data/model_results.csv")
print("\nTraining completed successfully.")