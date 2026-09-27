import joblib
import pandas as pd

MODEL_PATH = "models/machine_failure_model.pkl"

model = joblib.load(MODEL_PATH)

sample = pd.DataFrame({
    "Type": ["M"],
    "Air temperature [K]": [300.0],
    "Process temperature [K]": [310.0],
    "Rotational speed [rpm]": [1500],
    "Torque [Nm]": [40.0],
    "Tool wear [min]": [100]
})

prediction = model.predict(sample)[0]
probability = model.predict_proba(sample)[0][1]

print("Machine Failure Prediction:", prediction)
print(
    "Failure Probability:",
    f"{probability * 100:.2f}%"
)

if prediction == 1:
    print("Status: FAILURE RISK")
else:
    print("Status: NORMAL")