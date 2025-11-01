











import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- Load Model and Encoders ---
try:
    # Note: We load 'customer_churn_model.pkl' as saved in your notebook
    with open("customer_churn_model.pkl", "rb") as f:
        model_data = pickle.load(f)
    model = model_data['model']
except FileNotFoundError:
    st.error("Model file 'customer_churn_model.pkl' not found. Please run the notebook to generate it.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

try:
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
except FileNotFoundError:
    st.error("Encoder file 'encoders.pkl' not found. Please run the notebook to generate it.")
    st.stop()
except Exception as e:
    st.error(f"Error loading encoders: {e}")
    st.stop()


# --- Page Configuration ---
st.set_page_config(page_title="Customer Churn Prediction", page_icon="👋")
st.title("👋 Customer Churn Prediction")
st.write("This app predicts whether a customer will churn based on their account information. Please fill in the details below.")

# --- Input Form ---
# Get the unique values from your notebook's EDA to populate dropdowns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Details")
    gender = st.selectbox("Gender", ('Male', 'Female'))
    senior_citizen = st.selectbox("Senior Citizen", ('No', 'Yes'))
    partner = st.selectbox("Partner", ('No', 'Yes'))
    dependents = st.selectbox("Dependents", ('No', 'Yes'))
    tenure = st.slider("Tenure (Months)", 0, 72, 12) # Min/Max from notebook describe()

with col2:
    st.subheader("Service Details")
    phone_service = st.selectbox("Phone Service", ('No', 'Yes'))
    multiple_lines = st.selectbox("Multiple Lines", ('No', 'Yes', 'No phone service'))
    internet_service = st.selectbox("Internet Service", ('DSL', 'Fiber optic', 'No'))
    online_security = st.selectbox("Online Security", ('No', 'Yes', 'No internet service'))
    online_backup = st.selectbox("Online Backup", ('No', 'Yes', 'No internet service'))

# Create more columns for the rest of the inputs to keep it clean
col3, col4 = st.columns(2)

with col3:
    device_protection = st.selectbox("Device Protection", ('No', 'Yes', 'No internet service'))
    tech_support = st.selectbox("Tech Support", ('No', 'Yes', 'No internet service'))
    streaming_tv = st.selectbox("Streaming TV", ('No', 'Yes', 'No internet service'))
    streaming_movies = st.selectbox("Streaming Movies", ('No', 'Yes', 'No internet service'))

with col4:
    st.subheader("Billing Details")
    contract = st.selectbox("Contract", ('Month-to-month', 'One year', 'Two year'))
    paperless_billing = st.selectbox("Paperless Billing", ('No', 'Yes'))
    payment_method = st.selectbox("Payment Method", ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=120.0, value=70.0, step=0.01)
    total_charges = st.number_input("Total Charges", min_value=0.0, max_value=9000.0, value=1500.0, step=0.01)


# --- Prediction Logic ---
if st.button("🚀 Predict Churn", use_container_width=True):
    # 1. Collect input data into a dictionary
    # Keys must match the feature names from your notebook
    input_data = {
        'gender': gender,
        'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0, # Map Yes/No to 1/0 as in notebook
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

    # 2. Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # 3. Apply the saved encoders
    # Get the list of columns to encode from the encoders.pkl file
    categorical_cols = list(encoders.keys())
    
    try:
        for col in categorical_cols:
            if col in input_df.columns:
                le = encoders[col]
                # Use .transform() on the loaded encoder
                input_df[col] = le.transform(input_df[col])
            else:
                st.warning(f"Column '{col}' not found in input data.")
                
        # 4. Make prediction
        prediction = model.predict(input_df)
        proba = model.predict_proba(input_df)
        
        churn_probability = proba[0][1] # Probability of 'Churn' (class 1)

        # 5. Display result
        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.error(f"**Customer is LIKELY to Churn**")
            st.metric(label="Churn Probability", value=f"{churn_probability:.1%}")
        else:
            st.success(f"**Customer is NOT likely to Churn**")
            st.metric(label="Churn Probability", value=f"{churn_probability:.1%}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.write("Please ensure all inputs are correct.")
