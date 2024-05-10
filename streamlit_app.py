import streamlit as st
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


def display_model_performance(model_name, fold):
    # Load model performance metrics
    params_filename = f"{model_name} fold_{fold}_params.pkl"
    with open(params_filename, 'rb') as f:
        params = pickle.load(f)

    # Display performance metrics
    st.write(f"Model: {model_name}, Fold: {fold}")
    st.write("Best Parameters:", params)
    # Add more performance metrics as needed


# Main Streamlit app logic
def main():
    # Title and description
    st.title("Model Performance Analysis")
    st.write("This app displays the performance metrics of various models.")

    # Select model and fold
    model_list = ["Logistic Regression", "Random Forest", "XGBoost", "DNN"]
    selected_model = st.selectbox("Select Model", model_list)
    selected_fold = st.selectbox("Select Fold", [3, 4, 5, 10])

    # Display model performance metrics
    if selected_model == "Logistic Regression":
        display_model_performance("Logistic Regression", selected_fold)
    elif selected_model == "Random Forest":
        display_model_performance("Random Forest", selected_fold)
    elif selected_model == "XGBoost":
        display_model_performance("XGBoost", selected_fold)
    elif selected_model == "DNN":
        display_model_performance("DNN", selected_fold)


if __name__ == "__main__":
    main()
