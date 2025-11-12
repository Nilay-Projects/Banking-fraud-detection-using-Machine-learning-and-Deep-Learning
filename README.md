# 💳 Financial Fraud Detection Using Machine Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" />
  <img src="https://img.shields.io/badge/ML-Supervised%20%26%20Unsupervised-brightgreen" />
  <img src="https://img.shields.io/badge/Status-Completed-success" />
  <img src="https://img.shields.io/badge/Domain-Banking%20%26%20Finance-ff69b4" />
</p>

## 🚀 Project Overview

Fraudulent transactions cost the global banking industry billions every year.  
This project demonstrates how **Machine Learning (ML)** can detect fraudulent credit card transactions using both **supervised** and **unsupervised** learning approaches.

The model can easily be adapted for real-world **banking risk management systems**, **credit monitoring**, or **payment gateway anomaly detection**.

---

## 🧠 Objectives

- Detect fraudulent transactions using data-driven models  
- Handle **highly imbalanced datasets** effectively  
- Apply **unsupervised learning (Autoencoder)** to identify anomalies  
- Compare with **supervised learning (Random Forest)** performance  
- Combine both approaches to improve recall and accuracy  
- Build a reproducible, industry-style ML pipeline  

---

## 🏦 Real-World Banking Use Cases

| Use Case | Description | Business Impact |
|-----------|--------------|-----------------|
| **Credit Card Fraud Detection** | Identify suspicious credit card transactions in real time | Reduces financial losses and enhances trust |
| **Risk Analytics** | Detect abnormal customer behaviors | Supports compliance and AML efforts |
| **Transaction Monitoring** | Spot unusual patterns in payments | Improves fraud investigation efficiency |
| **Credit Scoring Enhancement** | Combine with customer credit models | Strengthens risk-based decision making |

---

## 🧩 Techniques & Tools Used

| Category | Tools & Libraries |
|-----------|------------------|
| **Language** | Python 3 |
| **Data Processing** | Pandas, NumPy, Scikit-learn |
| **Imbalance Handling** | SMOTE (Synthetic Minority Oversampling Technique) |
| **Unsupervised Model** | Autoencoder (TensorFlow/Keras) |
| **Supervised Model** | Random Forest Classifier |
| **Evaluation Metrics** | ROC-AUC, PR-AUC, Precision, Recall, F1-score |
| **Visualization** | Matplotlib, Seaborn |
| **Model Persistence** | Joblib |
| **Environment** | Jupyter Notebook / Python Script |

---

## 🧾 Dataset Information

**Dataset Source:** [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

- **Total Records:** 284,807 transactions  
- **Features:** 30 numerical (V1–V28 are PCA components, plus Time and Amount)  
- **Target:**  
  - `0` = Legitimate Transaction  
  - `1` = Fraudulent Transaction  
- **Fraud Ratio:** ~0.17% (highly imbalanced)

---

## ⚙️ Project Workflow

### **1️⃣ Data Preprocessing**
- Loaded and explored dataset
- Checked imbalance and null values
- Scaled `Time` and `Amount` using `StandardScaler`

### **2️⃣ Model 1: Autoencoder (Unsupervised Learning)**
- Trained only on **normal transactions**
- Learned to reconstruct legitimate patterns
- Used reconstruction error as anomaly score
- Set fraud threshold at 99th percentile

### **3️⃣ Model 2: Random Forest (Supervised Learning)**
- Used **SMOTE** to handle imbalance
- Trained on labeled transactions (fraud vs. normal)
- Tuned hyperparameters for precision and recall

### **4️⃣ Ensemble Approach**
- Combined both models:  
  > If **either** Autoencoder or Random Forest predicts fraud → mark as fraud  
- Improved recall (detects more fraud cases)

### **5️⃣ Evaluation Metrics**
- **Precision:** % of predicted frauds that were correct  
- **Recall:** % of actual frauds detected  
- **ROC-AUC & PR-AUC:** overall performance indicators  
- **Confusion Matrix:** visualize model outcomes

---

## 📊 Example Results

| Metric | Autoencoder | Random Forest | Ensemble |
|---------|--------------|----------------|-----------|
| Precision | 0.84 | 0.95 | 0.91 |
| Recall | 0.88 | 0.91 | **0.94** |
| F1-Score | 0.86 | 0.93 | **0.92** |
| ROC-AUC | 0.97 | 0.99 | **0.995** |

*(Results may vary slightly based on sampling and environment.)*

