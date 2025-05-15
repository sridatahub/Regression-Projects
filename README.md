# 🏠 House Price Prediction Model

A machine learning model that predicts house prices based on features from the California housing dataset using XGBoost Regressor.
## 📁 Dataset Description

This project uses the **California Housing Dataset**, derived from the 1990 U.S. Census. It contains housing data for block groups — the smallest units used by the U.S. Census Bureau (typically 600–3,000 people).

- **Instances**: 20,640  
- **Features**: 8 numeric predictors  
- **Target**: Median house value (`MedHouseVal`) in $100,000s  
- **Missing Values**: None  
- **Source**: [UCI Repository](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html)

---

### 🔍 Feature Overview

| Feature      | Description |
|--------------|-------------|
| **MedInc**   | Median income in the block group (in tens of thousands). |
| **HouseAge** | Median age of houses in the area. |
| **AveRooms** | Average number of rooms per household. |
| **AveBedrms**| Average number of bedrooms per household. |
| **Population**| Total population in the block group. |
| **AveOccup** | Average number of people per household. |
| **Latitude** | Geographic latitude of the block group. |
| **Longitude**| Geographic longitude of the block group. |

---

### 🎯 Target Variable

- **MedHouseVal**: Median house value per block group, expressed in **$100,000s** — this is the value predicted by the model.
### 🔧 Project Workflow

1. Importing Dependencies & Dataset  
2. Creating DataFrame  
3. Data Exploration  
4. Correlation Analysis  
5. Data Splitting  
6. Model Training & Evaluation

### 📦 Importing Dependencies

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split

# Install XGBoost (if not already installed)
!pip install xgboost

import xgboost
from xgboost import XGBRegressor
from sklearn import metrics
```
### 📥 Loading the Dataset
```python
from sklearn.datasets import fetch_california_housing
dataset = fetch_california_housing()
dataset
```
### 📥 Creating Pandas DataFrame

```python
import pandas as pd
dataframe = pd.DataFrame(dataset.data, columns=dataset.feature_names)
dataframe.head()

#Adding Label (Price) to the DataFrame
dataframe['Houseprice'] = dataset.target
dataframe.head()
```
### 🔍 Data Exploration

```python
# Checking the number of rows and columns in the DataFrame
print(dataframe.shape)

# Checking for missing values
print(dataframe.isnull().sum())

# Looking at statistical measures of the dataset
print(dataframe.describe())
```
### 🔗 Correlation Analysis

Correlation measures the statistical relationship between two variables, indicating how one variable may change when the other does. It helps identify patterns and associations in data.

**Types of Correlation:**

- **Positive Correlation:** Both variables increase or decrease together (e.g., house size and price).
- **Negative Correlation:** One variable increases while the other decreases (e.g., distance from city center and house price).
#Understanding correlation among featues
```python
correlation = dataframe.corr()
```
### 🔥 Heatmap

A heatmap is a graphical representation of data where individual values in a matrix are represented as colors. In correlation analysis, heatmaps help visualize the strength and direction of relationships between multiple variables at once, making it easier to spot strong positive or negative correlations.

```python
#Plotting Heatmap to see the correlation
import seaborn as sns
plt.figure(figsize=(6,6))
sns.heatmap(correlation,cbar=True,fmt='.1f',square=True,annot=True,annot_kws={'size':8},cmap='Blues')
```
### 📊 Data Splitting

```python
# Separating features and target label
features = dataframe.drop(['Houseprice'], axis=1)
label = dataframe['Houseprice']
print(features)
print(label)

# Splitting the data into training and testing sets
from sklearn.model_selection import train_test_split

# Splitting the dataset into 80% training and 20% testing data to evaluate model performance on unseen data
X_train, x_test, Y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=2)
print(features.shape, X_train.shape, x_test.shape)
print(label.shape, Y_train.shape, y_test.shape)
```
### 🤖 Model Training & Evaluation

```python
from xgboost import XGBRegressor
from sklearn import metrics
import matplotlib.pyplot as plt

# Initialize the model
model = XGBRegressor()

# Train the model on training data
model.fit(X_train, Y_train)

# Predict on training data
training_data_prediction = model.predict(X_train)
# Calculate evaluation metrics
r_score = metrics.r2_score(Y_train, training_data_prediction)
mae = metrics.mean_absolute_error(Y_train, training_data_prediction)

print("Training R Square error:", r_score)
print("Training Mean Absolute error:", mae)

# Visualization: Actual vs Predicted Prices (Training Data)
plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Training Data: Actual vs Predicted Prices")
plt.show()

test_data_prediction = model.predict(x_test)

r_score_test = metrics.r2_score(y_test, test_data_prediction)
mae_test = metrics.mean_absolute_error(y_test, test_data_prediction)

print("Testing R Square error:", r_score_test)
print("Testing Mean Absolute error:", mae_test)
```
### 📊 Results

**Training Accuracy (R² Score):** 0.9437  
**Mean Absolute Error:** 0.1934

**Testing Accuracy (R² Score):** 0.8338  
**Mean Absolute Error:** 0.3109
Overall, the model performs quite well in predicting house prices with good accuracy on both training and testing data.







