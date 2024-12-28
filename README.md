# Comparative Analysis of Machine Learning and Deep Learning Models for Car Price Prediction

## Overview
This project focuses on predicting car prices by comparing various machine learning and deep learning models. It evaluates model performance to determine the most effective approach for accurate car price prediction.

---

## Dataset
- **Source:** Kaggle
- **Attributes:**
  - `Car_Name`: Name of the car.
  - `Year`: Year of manufacture.
  - `Selling_Price`: Price at which the car was sold.
  - `Present_Price`: Current ex-showroom price of the car.
  - `Kms_Driven`: Total kilometers driven by the car.
  - `Fuel_Type`: Type of fuel used (Petrol/Diesel/Other).
  - `Seller_Type`: Type of seller (Individual/Dealer).
  - `Transmission`: Type of transmission (Manual/Automatic).
  - `Owner`: Number of previous owners.

---

## Features
- **Exploratory Data Analysis (EDA):**
  - Visualizing data distributions.
  - Correlation analysis using heatmaps.
- **Feature Engineering:**
  - Adding derived features like `Car_Age`.
  - Encoding categorical variables.
- **Data Scaling:**
  - Standardizing numerical features for better model performance.

---

## Technologies Used
- **Programming Language:** Python
- **Libraries and Tools:**
  - Pandas, NumPy
  - Matplotlib, Seaborn
  - Scikit-learn, XGBoost
  - TensorFlow, Keras
  - Jupyter Notebook

---

## Project Workflow
1. Load and preprocess the dataset.
2. Visualize data distributions and relationships.
3. Train machine learning models:
   - Linear Regression
   - Decision Tree
   - Random Forest
   - Support Vector Regressor (SVR)
   - k-Nearest Neighbors (KNN)
   - XGBoost
4. Implement and train a deep learning model using TensorFlow/Keras.
5. Evaluate models using metrics like Mean Squared Error (MSE) and R-squared score.

---

## Model Comparisons
| Model                        | Mean Squared Error | R-squared Score |
|------------------------------|--------------------|-----------------|
| Linear Regression            | 3.7578             | 0.8291          |
| Decision Tree Regressor      | 1.1609             | 0.9480          |
| Random Forest Regressor      | 1.1268             | 0.9539          |
| Support Vector Regressor     | 4.0447             | 0.9539          |
| k-Nearest Neighbors Regressor| 3.6691             | 0.8420          |
| XGBoost Regressor            | 0.9739             | 0.9578          |
| Deep Learning Model          | 0.5042             | 0.9811          |

---


## Results
- **Best Performing Model:** Artificial Neural Network

