# Rossmann Store Sales Forecasting

## 🛒 Project Overview

Rossmann is the largest drug store chain in Germany, operating over 3,000 stores across Europe. This project focuses on building a predictive model to forecast daily sales for each Rossmann store, accounting for factors such as promotions, competition, holidays, and seasonality.

The goal is to use historical data to predict sales for a future period, following the Kaggle competition format.

## 📁 Dataset Description

The dataset consists of four main files:

- **train.csv**: Historical daily sales data, including:
  - `Store`: store ID
  - `DayOfWeek`, `Date`, `Sales`, `Customers`
  - `Open`, `Promo`, `StateHoliday`, `SchoolHoliday`

- **test.csv**: Same structure as train.csv but missing `Sales` and `Customers`. This is the test set used for prediction.

- **store.csv**: Additional metadata about each store:
  - `StoreType`, `Assortment`, `CompetitionDistance`, `Promo2`, etc.

- **sample_submission.csv**: Template for submitting predictions to Kaggle.

## 🎯 Objectives

- Perform **exploratory data analysis (EDA)** to understand patterns and trends in the data.
- Clean and preprocess the data, handle **missing values**, and **engineer useful features**.
- Build a baseline model using **Decision Tree Regressor** and evaluate using **RMSPE**.
- Train a more powerful **XGBoost model** with hyperparameter tuning to improve prediction accuracy.

## 🧪 Evaluation Metric

The primary evaluation metric is **Root Mean Square Percentage Error (RMSPE)**:

\[
\text{RMSPE} = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} \left( \frac{\hat{y}_i - y_i}{y_i} \right)^2 }
\]

Where:
- \( y_i \) = actual sales
- \( \hat{y}_i \) = predicted sales

Zero-sales days are excluded from evaluation.

## 🛠️ Project Workflow

### Step 1: Load the Data
Use `pandas` to load and inspect `train.csv`, `test.csv`, `store.csv`.

### Step 2: Exploratory Data Analysis (EDA)
Use `matplotlib`, `seaborn`, and `pandas` profiling to:
- Visualize sales trends over time
- Compare open/closed days, promotions, holidays
- Examine missing values and distribution of key features

### Step 3: Data Preprocessing
- Handle missing values (drop/fill/flag)
- Convert dates to datetime format and extract features (year, month, day, weekday)
- Encode categorical features

### Step 4: Feature Engineering
- Extract time-based and store-based features
- Merge `store.csv` with `train.csv` and `test.csv`

### Step 5: Baseline Model
- Train a **DecisionTreeRegressor** with cross-validation
- Tune `max_depth` using `GridSearchCV`
- Evaluate using **neg_RMSPE**

### Step 6: XGBoost Model
- Train an **XGBoostRegressor**
- Key hyperparameters:
  - `eta` (learning rate)
  - `max_depth`
  - `subsample`
  - `colsample_bytree`
  - `num_trees`
- Perform parameter tuning and cross-validation

## 🧠 Tools and Libraries

- Python 3.x
- Pandas
- NumPy
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost
- Jupyter Notebook

## 🚀 How to Run

1. Clone the repo and place the dataset files (`train.csv`, `test.csv`, `store.csv`, `sample_submission.csv`) in the working directory.
2. Open `Rossmann_Sales_Forecasting.ipynb` (or your notebook file).
3. Follow the notebook cells to preprocess data, build models, and generate predictions.
4. Save predictions using:
   ```python
   submission.to_csv("submission.csv", index=False)
