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

### 📥 Loading the Dataset

```python
from sklearn.datasets import fetch_california_housing
dataset = fetch_california_housing()
dataset

import pandas as pd
dataframe = pd.DataFrame(dataset.data, columns=dataset.feature_names)
dataframe.head()



