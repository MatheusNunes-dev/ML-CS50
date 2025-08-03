# 🤖 Churn Prediction Model – Machine Learning Project

This repository contains the **machine learning pipeline** and analysis for a churn prediction problem based on an e-commerce customer dataset. The project was developed using knowledge from the **Teo Me Why - Machine Learning course**, focusing on real-world problem-solving and solid data science practices.

## 🧠 Objective

To develop a machine learning model capable of predicting whether a customer will **churn (leave)** an e-commerce platform, based on behavioral and demographic data.

## 📊 Dataset

- Source: [Kaggle E-Commerce Churn Dataset](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction)
- Features include: last visit date, purchase history, customer satisfaction score, frequency of use, and more.

## 🔨 ML Pipeline

The machine learning process includes:

- Data cleaning and preprocessing
- Feature engineering
- Train-test split
- Model training and validation
- Performance evaluation using metrics such as Accuracy, Precision, Recall, and F1 Score

📌 The **Future Engine** library was used to streamline **one-hot encoding**, while the rest of the pipeline was implemented using `scikit-learn`.

## 🔍 Models Tested

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

## ✅ Best Model Performance ( Test )

| Metric     | Score     |
|------------|-----------|
| Accuracy   | 0.896     |
| Precision  | 0.744     |
| Recall     | 0.593     |
| F1 Score   | 0.660     |
| AUC        | 0.898     |

## ✅ Best Model Performance ( OOT )

| Metric     | Score     |
|------------|-----------|
| Accuracy   | 0.914     |
| Precision  | 0.740     |
| Recall     | 0.606     |
| F1 Score   | 0.666     |
| AUC        | 0.889     |




## 🧰 Tools & Libraries

- Python
- Pandas
- Scikit-learn
- Matplotlib & Seaborn (visualization)
- Future Engine (only for OneHotEncoder usage)
- Jupyter Notebook

## 💡 Final Notes

This project was part of a broader initiative where the machine learning model was later integrated into a web application (see my [CS50 final project repository](link-to-your-repo)) that allows businesses to upload customer data and receive churn predictions with visual insights.
