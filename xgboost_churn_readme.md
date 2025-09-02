# 🏦 Customer Churn Prediction (XGBoost)

## 📌 Project Overview

This project predicts **customer churn** (whether a customer leaves the bank) using **XGBoost**.  
The model is trained on customer demographic and financial data and outputs churn risk.

### 🔑 Key Steps:
1. Data Cleaning & Preprocessing  
2. Exploratory Data Analysis (EDA + Visuals)  
3. Feature Engineering  
4. Model Training & Evaluation  
5. Interpretation of Results  
6. Saving Model for Deployment  

---

## 📊 Dataset

- **Source:** [Bank Customer Churn Dataset (Kaggle)](https://www.kaggle.com/datasets/shubhammeshram579/bank-customer-churn-prediction)  
- **Target:** `Exited` → (1 = Churned, 0 = Stayed)  
- **Features:** Demographics, account information, and financial data.  

---

## ⚙️ Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/lashm4/Bank-Churn-Prediction.git
cd Bank-Churn-Prediction
pip install -r requirements.txt
```

---

## 🚀 Usage

Run the Jupyter Notebook:

```bash
jupyter notebook notebooks/xgboost_churn_model.ipynb
```

Or load the pre-trained model for predictions:

```python
import joblib
import pandas as pd

# Load XGBoost model
xgb_model = joblib.load("models/xgboost_churn_model.pkl")

# Example usage
X_new = pd.DataFrame([[600, 40, 3, 60000]],
                     columns=["CreditScore","Age","NumOfProducts","EstimatedSalary"])
xgb_pred = xgb_model.predict(X_new)
print("XGBoost Churn Prediction:", xgb_pred)
```

---

## 📈 Model Performance

- **Test Accuracy:** 86.8%  
- **ROC-AUC:** 0.728 → good separation of churners  
- **Confusion Matrix:** [[1535, 58], [207, 201]]  
- **Classification Report:**
  - Class 0 (Non-churners): Precision = 0.88, Recall = 0.96, F1-score = 0.92  
  - Class 1 (Churners): Precision = 0.78, Recall = 0.49, F1-score = 0.60  

---

## 🔑 Key Features & Insights

Top features contributing to churn (XGBoost Feature Importance):
- NumOfProducts → 28.0%  
- Age → 17.2%  
- IsActiveMember → 16.5%  
- Geography_Germany → 11.4%  
- Gender → 6.1%  
- Balance → 6.0%  
- Geography_Spain → 4.3%  

Insights:
- Customers with fewer products, older age, and inactive status are more likely to churn.
- German customers have a higher risk of churn.

---

## 📌 Next Steps

- Compare XGBoost with Logistic Regression and other models (Random Forest, Gradient Boosting)
- Tune hyperparameters for better performance
- Deploy model via Streamlit or Flask for real-time predictions
- Monitor model performance on new data

---

## 📜 Requirements

Dependencies are listed in `requirements.txt`, including:
- pandas  
- numpy  
- scikit-learn  
- xgboost  
- seaborn  
- matplotlib  
- joblib  
- jupyter  

---

## 👩‍💻 Author

Created by **Lashmi M.** – feel free to reach out!

