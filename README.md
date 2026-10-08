# 🫐 Wild Blueberry Yield Prediction

A machine learning regression project that predicts **wild blueberry crop yield** using pollination, weather, and fruit-development features.

## 📌 Project Overview

The project includes:

- Exploratory Data Analysis
- Correlation analysis
- Feature engineering
- Regression model comparison
- Residual analysis
- Feature importance

## 📊 Dataset

The dataset contains:

```text
15,289 rows
18 original columns
```

Main features include:

- Bee activity
- Temperature ranges
- Rainfall
- Fruit set
- Fruit mass
- Seeds

Target:

```text
yield
```

No missing values are reported.

## 🧹 Feature Engineering

The notebook creates:

- Total bees
- Bee diversity
- Temperature features
- Bee × rain impact
- Fruit mass × fruit set
- Fruit mass × seeds
- Seeds per fruit mass

## 🤖 Models

The project compares several regression models.

Best model:

```text
Gradient Boosting Regressor
R²   ≈ 0.824
RMSE ≈ 556.19
MAE  ≈ 349.99
```

## 📈 Key Finding

`fruitset`, `seeds`, and `fruitmass` show the strongest relationships with blueberry yield.

## 🛠️ Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Yellowbrick
- Joblib
- Jupyter Notebook

## 🎯 Skills Demonstrated

- Regression Modeling
- Feature Engineering
- Model Comparison
- Residual Analysis
- Feature Importance
- Agricultural Data Analysis

---

Built with Python, Scikit-learn, and Gradient Boosting. 🫐📊🤖
