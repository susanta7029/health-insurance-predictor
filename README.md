# 💰 Health Insurance Cost Prediction

A data science project that predicts medical insurance charges based on individual features like age, gender, BMI, smoking status, number of children, and region using regression models.

---

## 📌 Project Overview

This project aims to:
- Analyze the factors affecting medical insurance charges
- Build and evaluate regression models
- Deploy a web app using Streamlit for real-time prediction

---

## 📊 Dataset

- Source: [Kaggle - Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- Features:
  - `age`: Age of the person
  - `sex`: Gender
  - `bmi`: Body Mass Index
  - `children`: Number of dependents
  - `smoker`: Smoking status
  - `region`: Residential area
  - `charges`: Medical insurance cost (target variable)

---

## ⚙️ Technologies Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn (EDA)
- Scikit-learn (ML models)
- Streamlit (Web app)
- Render (Deployment)

---

## 📈 Models & Performance

| Model             | MAE     | RMSE    | R² Score |
|------------------|---------|---------|----------|
| Linear Regression| ~X      | ~Y      | ~Z       |
| Random Forest     | ~X      | ~Y      | ~Z       |

> Random Forest outperformed Linear Regression with better R² and lower error values.

---

## 📊 Feature Importance

![feature_importance](feature_importance.png)  
*Smoking status, age, and BMI are top predictors of insurance cost.*

---

## 🚀 Live Demo

🖥️ Streamlit Web App: [Click to Try It](https://your-render-url.onrender.com)

---

## 💡 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/your-username/health-insurance-predictor.git

# Navigate into the project
cd health-insurance-predictor

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run insurance_app.py
