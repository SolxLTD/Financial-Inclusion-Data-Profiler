# 🧠 Financial Inclusion Prediction using Machine Learning

## 📘 Project Overview
This project focuses on predicting **whether an individual owns a bank account** using demographic and socio-economic data.  
The goal is to support **financial inclusion efforts**, which aim to ensure access to affordable financial services for all individuals.

This is a **machine learning classification project**, developed and deployed as a **Streamlit web application**.

---

## 📊 Dataset Description
The dataset used in this project was **collected locally** (not from Kaggle) and contains responses from multiple individuals across various regions.  
It includes demographic, household, and economic variables relevant to financial inclusion.

### Key Features
| Column | Description |
|--------|--------------|
| `country` | Country of respondent |
| `year` | Year of survey |
| `uniqueid` | Unique identifier for each respondent |
| `bank_account` | Target variable — whether respondent has a bank account (Yes/No) |
| `location_type` | Urban or Rural |
| `cellphone_access` | Ownership of cellphone (Yes/No) |
| `household_size` | Number of people in the household |
| `age_of_respondent` | Age of respondent |
| `gender_of_respondent` | Male or Female |
| `relationship_with_head` | Relationship to head of household |
| `marital_status` | Marital status |
| `education_level` | Highest level of education attained |
| `job_type` | Type of employment |

---

## ⚙️ Project Workflow

### 1. Data Exploration & Profiling
- Displayed general dataset information (shape, columns, data types).
- Generated a **Pandas Profiling Report** using `ydata_profiling`.
- Checked and handled **missing values**, **duplicates**, and **outliers**.
- Converted categorical variables using **OneHotEncoding**.

### 2. Data Preprocessing
- Dropped irrelevant columns (`uniqueid`).
- Encoded categorical variables numerically.
- Scaled numeric features using `StandardScaler`.
- Split the dataset into **train** and **test** sets (80/20).

### 3. Model Selection
Two models were evaluated:
- Logistic Regression ✅ *(final model chosen)*
- Random Forest Classifier

The **Logistic Regression** model was selected because:
- It performed consistently well with interpretable coefficients.
- It is robust with relatively small datasets.
- It aligns with the binary classification nature of the problem.

### 4. Model Evaluation
The model was evaluated using:
- **Accuracy Score**
- **Classification Report**
- **Confusion Matrix**

---

## 🧪 Results Summary
| Metric | Score |
|--------|--------|
| Accuracy | ~0.85 |
| Precision & Recall | Balanced |
| Confusion Matrix | Clear distinction between classes |

---

## 💻 Streamlit Application

### Features
- Upload or preview the dataset  
- View profiling report directly in the browser  
- Remove duplicates and outliers interactively  
- Train and evaluate models in real-time  
- View confusion matrix and classification report  
- Make predictions using form inputs  

### How to Run Locally
```bash
git clone https://github.com/yourusername/financial-inclusion-ml.git
cd financial-inclusion-ml
python -m venv venv
venv\Scripts\activate     # (Windows)
pip install -r requirements.txt
streamlit run streamlit_financial_inclusion_app.py
