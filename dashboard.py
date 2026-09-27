import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="Quantum-Inspired Cloud IDS",
    page_icon="🛡️",
    layout="wide"
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    model = joblib.load(
        "models/quantum_multiclass_model.pkl"
    )

    return model


model = load_model()


# --------------------------------------------------
# LOAD SELECTED FEATURES
# --------------------------------------------------

with open(
    "models/quantum_multiclass_selected_features.txt",
    "r"
) as file:

    selected_features = [
        line.strip()
        for line in file.readlines()
    ]


# --------------------------------------------------
# LOAD REAL DATASET
# --------------------------------------------------

@st.cache_data
def load_dataset():

    file_path = (
        "data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    )

    data = pd.read_csv(file_path)

    data.columns = data.columns.str.strip()

    data = data.replace(
        [float("inf"), float("-inf")],
        float("nan")
    )

    data = data.dropna()

    return data


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title(
    "🛡️ Quantum-Inspired Cyber Intrusion Detection"
)

st.write(
    "Intrusion Detection System for Enterprise Cloud Platforms"
)

st.divider()


# --------------------------------------------------
# PROJECT INFORMATION
# --------------------------------------------------

st.subheader("📊 Project Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Dataset",
        "CIC-IDS2017"
    )

with col2:
    st.metric(
        "Original Features",
        "78"
    )

with col3:
    st.metric(
        "Selected Features",
        "10"
    )

with col4:
    st.metric(
        "Feature Reduction",
        "87.18%"
    )


st.divider()


# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

st.subheader("🤖 Model Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Model",
        "Random Forest"
    )

with col2:
    st.metric(
        "Accuracy",
        "98.27%"
    )

with col3:
    st.metric(
        "Attack Classes",
        "15"
    )


st.divider()


# --------------------------------------------------
# SELECTED FEATURES
# --------------------------------------------------

st.subheader(
    "🔬 Quantum-Inspired Selected Features"
)

for number, feature in enumerate(
    selected_features,
    start=1
):

    st.write(
        f"{number}. {feature}"
    )


st.divider()


# ==================================================
# REAL DATASET TEST
# ==================================================

st.subheader(
    "🚨 Test Using Real CIC-IDS2017 Data"
)

st.write(
    "This section uses a real network record from "
    "the CIC-IDS2017 DDoS dataset."
)


if st.button(
    "🔴 Test Real DDoS Record",
    use_container_width=True
):

    data = load_dataset()

    # Get real DDoS records
    ddos_data = data[
        data["Label"] == "DDoS"
    ]

    # Take first real DDoS record
    sample = ddos_data.iloc[[0]]

    # Select required features
    input_data = sample[
        selected_features
    ]

    # Make prediction
    prediction = model.predict(
        input_data
    )[0]

    actual_label = sample[
        "Label"
    ].iloc[0]

    st.divider()

    st.subheader(
        "🔍 Real Dataset Test Result"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            "**Actual Label:**"
        )

        st.error(
            f"🔴 {actual_label}"
        )

    with col2:

        st.write(
            "**Model Prediction:**"
        )

        if prediction == "BENIGN":

            st.success(
                "🟢 BENIGN"
            )

        else:

            st.error(
                f"🔴 {prediction}"
            )

    st.write(
        "### Network Feature Values"
    )

    display_data = input_data.T

    display_data.columns = [
        "Value"
    ]

    st.dataframe(
        display_data,
        use_container_width=True
    )

    if prediction == actual_label:

        st.success(
            "✅ Prediction is CORRECT!"
        )

    else:

        st.warning(
            "⚠️ Prediction does not match the actual label."
        )


st.divider()


# ==================================================
# MANUAL INTRUSION TEST
# ==================================================

st.subheader(
    "🧪 Manual Intrusion Detection"
)

st.write(
    "Enter values for the 10 selected network features."
)


input_values = []


for feature in selected_features:

    value = st.number_input(
        feature,
        value=0.0
    )

    input_values.append(
        value
    )


if st.button(
    "🔎 Detect Intrusion",
    use_container_width=True
):

    input_data = pd.DataFrame(
        [input_values],
        columns=selected_features
    )

    prediction = model.predict(
        input_data
    )[0]

    st.divider()

    st.subheader(
        "Prediction Result"
    )

    if prediction == "BENIGN":

        st.success(
            "🟢 BENIGN — No attack detected"
        )

    else:

        st.error(
            f"🔴 ATTACK DETECTED: {prediction}"
        )


st.divider()


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.caption(
    "Quantum-Inspired Machine Learning for "
    "Cyber Intrusion Detection"
)