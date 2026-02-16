
# CUSTOMER CHURN PREDICTION - STREAMLIT DEPLOYMENT


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("📊 Customer Churn Prediction System")
st.markdown("""
This application predicts customer churn using a **Tuned Random Forest Model**
with engineered features and optimized threshold (**0.40**).
""")

# ----------------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------------
model = joblib.load("rf_model.pkl")
top_15_features = joblib.load("top15.pkl")

# ----------------------------------------------------------
# FILE UPLOAD
# ----------------------------------------------------------
st.header("📂 Upload Customer Data (CSV File)")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("🔎 Uploaded Data Preview")
    st.dataframe(df.head())

    # ------------------------------------------------------
    # DATA CLEANING
    # ------------------------------------------------------
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    replace_cols = [
        "MultipleLines","OnlineSecurity","OnlineBackup",
        "DeviceProtection","TechSupport",
        "StreamingTV","StreamingMovies"
    ]

    for col in replace_cols:
        if col in df.columns:
            df[col] = df[col].replace({
                "No internet service": "No",
                "No phone service": "No"
            })

    
    # FEATURE ENGINEERING
   

    # 1️⃣ TenureGroup
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[0,12,36,72],
        labels=["New","Mid","Loyal"]
    )

    # 2️⃣ TotalServices
    services = [
        "PhoneService","MultipleLines","OnlineSecurity",
        "OnlineBackup","DeviceProtection","TechSupport",
        "StreamingTV","StreamingMovies"
    ]
    df["TotalServices"] = (df[services] == "Yes").sum(axis=1)

    # 3️⃣ AutoPay
    df["AutoPay"] = df["PaymentMethod"].apply(
        lambda x: 1 if "automatic" in x else 0
    )

    # 4️⃣ AvgMonthlySpend
    df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)

    # 5️⃣ ContractRisk
    df["ContractRisk"] = df["Contract"].map({
        "Month-to-month":2,
        "One year":1,
        "Two year":0
    })

    st.success("Feature Engineering Completed Successfully")

    
    # DISPLAY ENGINEERED FEATURES
    
    st.header("🧠 Engineered Features (Preview)")

    engineered_cols = [
        "TenureGroup","TotalServices",
        "AutoPay","AvgMonthlySpend","ContractRisk"
    ]

    st.dataframe(df[engineered_cols].head())

    
    # ENCODING
    
    X = pd.get_dummies(df, drop_first=True)

    for col in top_15_features:
        if col not in X.columns:
            X[col] = 0

    X = X[top_15_features]

    
    # PREDICTION
    
    probabilities = model.predict_proba(X)[:,1]
    threshold = 0.40
    predictions = (probabilities > threshold).astype(int)

    df["Churn_Probability"] = probabilities
    df["Predicted_Churn"] = predictions

    
    # RESULTS SUMMARY
   
    st.header("📈 Prediction Summary")

    col1, col2, col3 = st.columns(3)

    total_customers = len(df)
    churn_count = df["Predicted_Churn"].sum()
    churn_rate = (churn_count / total_customers) * 100

    col1.metric("Total Customers", total_customers)
    col2.metric("Predicted Churners", churn_count)
    col3.metric("Churn Rate (%)", round(churn_rate,2))

    st.subheader("📊 Detailed Prediction Table")
    st.dataframe(df)

    
    # FEATURE IMPORTANCE 
  
    st.header("📊 Model Feature Importance")

    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": top_15_features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    col_left, col_center, col_right = st.columns([1,2,1])

    with col_center:
        fig1, ax1 = plt.subplots(figsize=(6,4))
        ax1.barh(importance_df["Feature"], importance_df["Importance"])
        ax1.invert_yaxis()
        ax1.set_title("Top 15 Feature Importance")
        st.pyplot(fig1)

    
    # RISK SEGMENTATION 
    
    st.header("📌 Risk Segmentation")

    df["Risk_Level"] = pd.cut(
        df["Churn_Probability"],
        bins=[0,0.4,0.7,1],
        labels=["Low Risk","Medium Risk","High Risk"]
    )

    risk_counts = df["Risk_Level"].value_counts()

    col_left, col_center, col_right = st.columns([1,2,1])

    with col_center:
        fig2, ax2 = plt.subplots(figsize=(4,4))
        ax2.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%')
        ax2.set_title("Customer Risk Segmentation")
        st.pyplot(fig2)

    
    # PROBABILITY DISTRIBUTION 
    
    st.header("📉 Churn Probability Distribution")

    col_left, col_center, col_right = st.columns([1,2,1])

    with col_center:
        fig3, ax3 = plt.subplots(figsize=(6,4))
        ax3.hist(df["Churn_Probability"], bins=10)
        ax3.set_xlabel("Churn Probability")
        ax3.set_ylabel("Count")
        ax3.set_title("Probability Distribution")
        st.pyplot(fig3)

    
    # SHOW SELECTED FEATURES
    
    st.header("🔍 Selected Features Used by Model")
    st.write(top_15_features)

    
    # DOWNLOAD RESULTS
    
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇ Download Prediction Results",
        data=csv,
        file_name="churn_predictions.csv",
        mime="text/csv"
    )

else:
    st.info("Please upload a CSV file to begin prediction.")
