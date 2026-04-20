"""Streamlit web app for the ML paradigm classifier."""

import streamlit as st

from ml_paradigm_classifier import predict_ml_type, train_model


@st.cache_resource
def get_model():
    """Train once and reuse the model while the Streamlit app is open."""

    model, _X_train, _X_test, _y_train, _y_test = train_model()
    return model


st.set_page_config(page_title="ML Paradigm Classifier", page_icon="ML")

st.title("ML Paradigm Classifier")
st.write(
    "Type a business problem and get a beginner-friendly suggestion for the "
    "most likely machine learning paradigm."
)

example = "Predict customer churn using past data"
user_text = st.text_area(
    "Business problem description",
    value=example,
    height=120,
)

if st.button("Classify"):
    if not user_text.strip():
        st.warning("Please enter a business problem description.")
    else:
        model = get_model()
        result = predict_ml_type(model, user_text)

        st.subheader("Result")
        st.metric("ML Type", result.ml_type.title())
        st.metric("Confidence", f"{result.confidence:.2%}")

        st.write("**Reason**")
        st.info(result.reason)

        st.write("**Suggested Algorithms**")
        for algorithm in result.suggested_algorithms:
            st.write(f"- {algorithm}")

        st.write("**5-Step Workflow**")
        for step_number, step in enumerate(result.workflow, start=1):
            st.write(f"{step_number}. {step}")

        with st.expander("Model details"):
            st.write(f"Model prediction: {result.model_prediction.title()}")
            st.write(f"Model confidence: {result.model_confidence:.2%}")
            st.write(f"Matched keyword: {result.matched_keyword or 'None'}")
