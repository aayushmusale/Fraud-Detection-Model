# Fraud Detection Model

## 📌 Project Overview

This project aims to build a machine learning model capable of detecting fraudulent financial transactions. Using a dataset of transaction records, the project involves Exploratory Data Analysis (EDA), data preprocessing, and the training of supervised learning models to classify transactions as legitimate or fraudulent.

The project addresses the challenge of identifying rare fraudulent events (0.13% of the data) within a large volume of legitimate transactions.

## 📂 Dataset Description

The dataset used (`Fraud.csv`) contains simulated transaction data. Key features include:

* **step**: Maps a unit of time in the real world (1 step = 1 hour).
* **type**: Type of transaction (CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER).
* **amount**: Amount of the transaction in local currency.
* **nameOrig**: Customer who started the transaction.
* **oldbalanceOrg**: Initial balance before the transaction.
* **newbalanceOrig**: New balance after the transaction.
* **nameDest**: Customer who is the recipient of the transaction.
* **oldbalanceDest**: Initial balance of the recipient before the transaction.
* **newbalanceDest**: New balance of the recipient after the transaction.
* **isFraud**: Target variable. Transactions made by fraudulent agents.
* **isFlaggedFraud**: Flags illegal attempts to transfer more than 200,000 in a single transaction.
  
## 🛠️ Technologies Used

* **Python**: Primary programming language.
* **Pandas & NumPy**: Data manipulation and numerical operations.
* **Matplotlib & Seaborn**: Data visualization.
* **Scikit-learn**: Data preprocessing, pipeline construction, and Logistic Regression model.
* **XGBoost**: Gradient boosting classifier for improved performance.
* **Joblib**: Model serialization.

## ⚙️ Methodology

### 1. Exploratory Data Analysis (EDA)

* Analyzed the distribution of transaction types (Payment, Transfer, Cash Out, etc.).
* Calculated the fraud rate, which was found to be approximately **0.13%**.
* Visualized fraud frequency by transaction type, revealing that fraud primarily occurs in `TRANSFER` and `CASH_OUT` transactions.

### 2. Preprocessing

A `ColumnTransformer` pipeline was implemented to prepare the data:

* **Numerical Features**: Scaled using `StandardScaler` (e.g., `step`, `amount`, `oldbalanceOrg`).
* **Categorical Features**: Encoded using `OneHotEncoder` (e.g., `type`).

### 3. Model Training

Two models were trained and evaluated:

* **Logistic Regression**: A baseline model configured with `class_weight='balanced'` to handle the severe class imbalance.
* **XGBoost Classifier**: An advanced ensemble model configured with parameters such as `n_estimators=100`,`max_depth=6`, `n_jobs=-1` and `scale_pos_weight=700` to capture non-linear patterns.

## 📊 Results

### Logistic Regression

* **Accuracy:** 95%
* **Precision (Fraud):** 0.02
* **Recall (Fraud):** 0.95
* **F1-Score (Fraud):** 0.05

**Classification Report:**

```
              precision    recall  f1-score   support

           0       1.00      0.95      0.97   1906322
           1       0.02      0.95      0.05      2464

    accuracy                           0.95   1908786
   macro avg       0.51      0.95      0.51   1908786
weighted avg       1.00      0.95      0.97   1908786

```

**Confusion Matrix:**

```
[[1809641,   96681],
 [    121,    2343]]

```

### XGBoost Classifier (Best Model)

The XGBoost model achieved significantly better performance, maintaining high recall while drastically improving precision.

* **Accuracy:** 99.98%
* **Precision (Fraud):** 0.98
* **Recall (Fraud):** 0.88
* **F1-Score (Fraud):** 0.93

**Classification Report:**

```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00   1906322
           1       0.98      0.88      0.93      2464

    accuracy                           1.00   1908786
   macro avg       0.99      0.94      0.96   1908786
weighted avg       1.00      1.00      1.00   1908786

```

**Confusion Matrix:**

```
[[1903681    2641]
 [     42    2422]]

```

## 🚀 Usage

1. **Clone the repository**:
```bash
git clone <repository-url>

```


2. **Install dependencies**:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib

```


3. **Run the Notebook**:
Open `Task.ipynb` in Jupyter Notebook or Google Colab to execute the analysis and training steps.
4. **Load the Model**:
```python
import joblib
model = joblib.load('bestmodel_fraud_detection_pipeline.joblib')
# model.predict(new_data)

```



## 📁 Project Structure

* `Task.ipynb`: The main Jupyter Notebook containing code for EDA, preprocessing, and modeling.
* `Fraud.csv`: The dataset file (ensure this is placed in the root directory).
* `Data Dictionary.txt`: Detailed description of dataset columns.
* `bestmodel_fraud_detection_pipeline.joblib`: The serialized best-performing model.
