
# Customer Churn Prediction
## 🌐 Live Demo

The model has been deployed using Hugging Face Spaces.

You can test the churn prediction system here:

🔗 https://huggingface.co/spaces/Harsshini/Feature_Engineering

For input use sample_input.csv uploaded in Git repo
## Problem Statement

Customer churn prediction helps businesses identify customers who are likely to discontinue a service. Early detection enables proactive retention strategies and reduces revenue loss.

This project builds and optimizes machine learning models to accurately predict customer churn using customer behavior, service usage, and contract-related features.

---

## Feature Engineering & Optimization

### 🔹 Feature Creation
The following high-impact features were engineered:

- **TenureGroup** – Customer lifecycle categorization (New, Mid, Loyal)
- **TotalServices** – Engagement score (number of services subscribed)
- **AutoPay** – Automatic payment indicator
- **AvgMonthlySpend** – Normalized spending intensity
- **ContractRisk** – Ordinal encoding based on churn risk

### 🔹 Feature Selection
- Used **Random Forest feature importance**
- Selected **Top 15 most important features**
- Reduced dimensionality to improve generalization

### 🔹 Class Imbalance Handling
- Used `class_weight='balanced'` (Logistic & Random Forest)
- Used `scale_pos_weight` (XGBoost)

### 🔹 Hyperparameter Tuning
Applied tuning to improve model performance:
- Random Forest (n_estimators, max_depth, min_samples_split, etc.)
- XGBoost (learning_rate, max_depth, subsample, colsample_bytree)

### 🔹 Threshold Analysis
- Tested multiple probability thresholds
- Selected **0.40** as optimal decision threshold
- Improved Recall and business alignment

---

## Models Implemented

- Logistic Regression
- Random Forest
- XGBoost

Models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

---

## Final Model Performance

**Selected Model:** Tuned Random Forest (Top 15 Features)  
**Optimized Threshold:** 0.40  

| Metric     | Value  |
|------------|--------|
| Accuracy   | 0.7274 |
| Precision  | 0.4913 |
| Recall     | 0.8387 |
| F1-Score   | 0.6197 |
| ROC-AUC    | 0.84   |

---

## Outcome

The final optimized model achieves strong recall while maintaining balanced precision, making it suitable for real-world churn prevention and customer retention strategies.

# ▶️ How to Run This Project

### 🔹 1. Install Required Libraries
Make sure the following libraries are installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
```

### 🔹 2. Run the Notebook
- Open `Feature_engineering_task.ipynb`
- Run all cells sequentially from top to bottom.

Ensure the dataset file:
Customer_churn_telco_dataset.csv

is present in the same directory.

---

### 🔹 3. Run Prediction on New Data
To test the model on unseen data:

- Place `sample_input.csv` in the same folder.
- Run the final prediction cell.
- The output will display:
  - Churn Probability
  - Predicted Churn (0 or 1)

---

### 🔹 4. Use Deployed Application (Optional)
You can also test the model using the deployed Hugging Face Space:

🔗 https://huggingface.co/spaces/Harsshini/Feature_Engineering









