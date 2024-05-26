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

# Define required features for each model
required_features = [
    'Avg_min_between_sent_tnx', 'Avg_min_between_received_tnx',
    'Time_Diff_between_first_and_last_(Mins)', 'Sent_tnx', 'Received_Tnx',
    'Number_of_Created_Contracts', 'max_value_received', 'avg_val_received',
    'avg_val_sent', 'total_Ether_sent', 'total_ether_balance',
    'ERC20_total_Ether_received', 'ERC20_total_ether_sent',
    'ERC20_total_Ether_sent_contract', 'ERC20_uniq_sent_addr.1',
    'ERC20_uniq_rec_token_name'
]

# Define function for processing data
def process_data(model, df):
    # Separate the Address and FLAG columns
    if 'Address' in df.columns:
        dfAddress = df['Address']
        df = df.drop(columns=['Address'])
    else:
        dfAddress = None

    if 'FLAG' in df.columns:
        df = df.drop(columns=['FLAG'])

    # Filter the dataframe to include only the required columns
    df = df[required_features]

    # Preprocess data
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    df_filled = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    df_scaled = pd.DataFrame(scaler.fit_transform(df_filled), columns=df.columns)


    print("df.shape: ", df.shape)

    # Predictions
    predictions = model.predict(df_scaled)

    # TODO: My implementation
    # Omit first two columns (Index, Address)
    # df = df.iloc[:, 1:]
    # categories = df.select_dtypes('O').columns.astype('category')
    # numericals = df.select_dtypes(include=['float', 'int']).columns

    # Drop the two categorical features
    # df.drop(df[categories], axis=1, inplace=True)

    # Replace missing values of numerical variables with median
    # df.fillna(df.median(), inplace=True)

    # Filtering the features with 0 variance
    # no_var = df.var() == 0
    # Drop features with 0 variance --- these features will not help in the performance of the model
    # df.drop(df.var()[no_var].index, axis=1, inplace=True)

    # drop = ['total_transactions_(including_tnx_to_create_contract)', 'ERC20_avg_val_rec',
    #         'ERC20_avg_val_rec', 'ERC20_max_val_rec', 'ERC20_min_val_rec', 'ERC20_uniq_rec_contract_addr',
    #         'max_val_sent', 'ERC20_avg_val_sent',
    #         'ERC20_min_val_sent', 'ERC20_max_val_sent', 'Unique_Sent_To_Addresses',
    #         'Unique_Received_From_Addresses', 'total_ether_received', 'ERC20_uniq_sent_token_name',
    #         'min_value_received', 'min_val_sent', 'ERC20_uniq_rec_addr']
    # df.drop(drop, axis=1, inplace=True)

    # drops = ['ERC20_uniq_sent_addr']
    # df.drop(drops, axis=1, inplace=True)

    # predictions = model.predict(df)

    return predictions, dfAddress

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
        predictions, dfAddress = process_data(model, df)

        # Combine predictions with original data
        if dfAddress is not None:
            df['Address'] = dfAddress
        df['Prediction'] = predictions

        # Display predictions
        st.write("Predictions:")
        st.write(df[['Address', 'Prediction']])

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
