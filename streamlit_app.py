import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pickle
from tensorflow.keras.models import model_from_json

# Define functions for loading models
def load_lr_model(fold):
    model_filename = f"Logistic Regression_fold_{fold}_model.pkl"
    with open(model_filename, 'rb') as f:
        lr_model = pickle.load(f)
    return lr_model

def load_rf_model(fold):
    model_filename = f"Random Forest_fold_{fold}_model.pkl"
    with open(model_filename, 'rb') as f:
        rf_model = pickle.load(f)
    return rf_model

def load_xgb_model(fold):
    model_filename = f"XGBoost_fold_{fold}_model.pkl"
    with open(model_filename, 'rb') as f:
        xgb_model = pickle.load(f)
    return xgb_model

def load_dnn_model(fold):
    model_architecture_filename = f"DNN_fold_{fold}_architecture.json"
    with open(model_architecture_filename, 'r') as f:
        dnn_model_architecture = f.read()
    model_weights_filename = f"DNN_fold_{fold}_weights.h5"
    dnn_model_weights = model_from_json(dnn_model_architecture)
    dnn_model_weights.load_weights(model_weights_filename)
    return dnn_model_weights

# Define function for processing data
def process_data(model, df):
    # Preprocess data
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    df_filled = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    df_scaled = pd.DataFrame(scaler.fit_transform(df_filled), columns=df.columns)

    # Predictions
    predictions = model.predict(df_scaled)
    return predictions

# Main Streamlit app logic
def main():
    # Title and description
    st.title("Model Selection and CSV Upload")
    st.write("This app allows you to upload a CSV file and select a model for prediction.")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    # Model selection
    model_list = ["Logistic Regression", "Random Forest", "XGBoost", "DNN"]
    selected_model = st.selectbox("Select Model", model_list)
    selected_fold = st.selectbox("Select Fold", [3, 4, 5, 10])

    if uploaded_file is not None:
        # Read the uploaded CSV file
        df = pd.read_csv(uploaded_file)

        # Display uploaded CSV data
        st.write("Uploaded CSV data:")
        st.write(df.head())


        # Load selected model
        if selected_model == "Logistic Regression":
            model = load_lr_model(selected_fold)
        elif selected_model == "Random Forest":
            model = load_rf_model(selected_fold)
        elif selected_model == "XGBoost":
            model = load_xgb_model(selected_fold)
        elif selected_model == "DNN":
            model = load_dnn_model(selected_fold)

        # Process data with selected model
        predictions = process_data(model, df)

        # Display predictions
        st.write("Predictions:")
        st.write(predictions)

        # Add download button for predictions
        csv_file = df.copy()
        csv_file["Prediction"] = predictions
        st.download_button(
            label="Download Predictions CSV",
            data=csv_file.to_csv().encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
