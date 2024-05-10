import streamlit as st
import pandas as pd
import pickle
import json
from tensorflow.keras.models import model_from_json


# Define functions for loading models and displaying results
def load_lr_model(fold):
    # Load Logistic Regression model
    model_filename = f"Logistic Regression_fold_{fold}_model.pkl"
    with open(model_filename, 'rb') as f:
        lr_model = pickle.load(f)
    return lr_model


def load_rf_model(fold):
    # Load Random Forest model
    model_filename = f"Random Forest_fold_{fold}_model.pkl"
    with open(model_filename, 'rb') as f:
        rf_model = pickle.load(f)
    return rf_model


def load_xgb_model(fold):
    # Load XGBoost model
    model_filename = f"XGBoost_fold_{fold}_model.pkl"
    with open(model_filename, 'rb') as f:
        xgb_model = pickle.load(f)
    return xgb_model


def load_dnn_model(fold):
    # Load DNN model architecture
    model_architecture_filename = f"DNN_fold_{fold}_architecture.json"
    with open(model_architecture_filename, 'r') as f:
        dnn_model_architecture = f.read()

    # Load DNN model weights
    model_weights_filename = f"DNN_fold_{fold}_weights.h5"
    dnn_model_weights = model_from_json(dnn_model_architecture)
    dnn_model_weights.load_weights(model_weights_filename)
    return dnn_model_weights


def process_data(model, df):
    # Process data with the selected model
    predictions = model.predict(df)
    return predictions


# Main Streamlit app logic
def main():
    # Title and description
    st.title("Fraud Detection Model")
    st.write("This app predicts whether transactions are fraudulent or not.")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the uploaded CSV file
        df = pd.read_csv(uploaded_file)

        # Display some information about the uploaded file
        st.write("Uploaded CSV file:")
        st.write(df.head())

        # Select model and fold
        model_list = ["Logistic Regression", "Random Forest", "XGBoost", "DNN"]
        selected_model = st.selectbox("Select Model", model_list)
        selected_fold = st.selectbox("Select Fold", [3, 4, 5, 10])

        # Load the selected model
        if selected_model == "Logistic Regression":
            model = load_lr_model(selected_fold)
        elif selected_model == "Random Forest":
            model = load_rf_model(selected_fold)
        elif selected_model == "XGBoost":
            model = load_xgb_model(selected_fold)
        elif selected_model == "DNN":
            model = load_dnn_model(selected_fold)

        # Process the data with the selected model
        predictions = process_data(model, df)

        # Display the predictions
        st.write("Predictions:")
        st.write(predictions)

        # Add a download button for the predictions
        csv_file = df.copy()
        csv_file["Prediction"] = predictions
        csv_file.to_csv("predictions.csv", index=False)
        st.download_button(
            label="Download Predictions CSV",
            data=csv_file.to_csv().encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
