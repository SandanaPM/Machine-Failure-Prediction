import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "models/machine_failure_model.pkl"

model = joblib.load(MODEL_PATH)

st.set_page_config(
    page_title="Machine Failure Prediction",
    page_icon="⚙️",
    layout="centered"
)

st.title("⚙️ Machine Failure Prediction")
st.write(
    "Predictive maintenance system using machine operating parameters."
)

st.info(
    "The model estimates the probability of machine failure "
    "based on the entered operating conditions."
)

st.sidebar.header("Machine Parameters")

product_type = st.sidebar.selectbox(
    "Product Type",
    ["L", "M", "H"]
)

air_temperature = st.sidebar.number_input(
    "Air Temperature [K]",
    min_value=250.0,
    max_value=350.0,
    value=300.0,
    step=0.1
)

process_temperature = st.sidebar.number_input(
    "Process Temperature [K]",
    min_value=250.0,
    max_value=400.0,
    value=310.0,
    step=0.1
)

rotational_speed = st.sidebar.number_input(
    "Rotational Speed [rpm]",
    min_value=500,
    max_value=5000,
    value=1500,
    step=10
)

torque = st.sidebar.number_input(
    "Torque [Nm]",
    min_value=0.0,
    max_value=100.0,
    value=40.0,
    step=0.1
)

tool_wear = st.sidebar.number_input(
    "Tool Wear [min]",
    min_value=0,
    max_value=300,
    value=100,
    step=1
)

st.subheader("Entered Machine Parameters")

col1, col2 = st.columns(2)

with col1:
    st.write("**Product Type:**", product_type)
    st.write("**Air Temperature:**", f"{air_temperature:.1f} K")
    st.write("**Process Temperature:**", f"{process_temperature:.1f} K")

with col2:
    st.write("**Rotational Speed:**", f"{rotational_speed} rpm")
    st.write("**Torque:**", f"{torque:.1f} Nm")
    st.write("**Tool Wear:**", f"{tool_wear} min")

if st.button(
    "Predict Machine Status",
    type="primary",
    use_container_width=True
):

    input_data = pd.DataFrame({
        "Type": [product_type],
        "Air temperature [K]": [air_temperature],
        "Process temperature [K]": [process_temperature],
        "Rotational speed [rpm]": [rotational_speed],
        "Torque [Nm]": [torque],
        "Tool wear [min]": [tool_wear]
    })

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(
        input_data
    )[0][1]

    st.divider()

    st.subheader("Prediction Result")

    st.metric(
        "Machine Failure Probability",
        f"{probability * 100:.2f}%"
    )

    if prediction == 1:
        st.error(
            "⚠️ Machine Failure Risk Detected"
        )
        st.warning(
            "The operating conditions indicate an elevated "
            "risk of machine failure."
        )
    else:
        st.success(
            "✅ Machine Operating Normally"
        )
        st.write(
            "The model does not classify the current operating "
            "conditions as a machine failure."
        )

st.divider()

st.caption(
    "Dataset: AI4I 2020 Predictive Maintenance Dataset "
    "from UCI Machine Learning Repository."
)