import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import opendatasets as od
import os

# Downloading the dataset
dataset_url = 'https://www.kaggle.com/datasets/altavish/boston-housing-dataset'
od.download(dataset_url)
data_dir = './boston-housing-dataset'
print(os.listdir(data_dir))

# Loading the dataset into a DataFrame
df = pd.read_csv(os.path.join(data_dir, 'HousingData.csv'))
print(df.head(10))

# Data cleaning and correcting
print(df.isnull().sum())
df.dropna(inplace=True)
print(df.isnull().sum())

print(df.info())
print(df.shape)
print(df.describe())

# Plotting histograms for numeric columns
df.hist(bins=20, figsize=(15, 10))
plt.show()

# Calculating and plotting the correlation matrix
plt.figure(figsize=(12, 12))
sns.heatmap(data=df.corr().round(2), annot=True, cmap='coolwarm', linewidths=0.2, square=True)
plt.show()

# Plotting pair plots for selected variables
sns.pairplot(df, vars=['RM', 'TAX', 'PTRATIO', 'LSTAT', 'MEDV'])
plt.show()

# Creating box plots
fig, ax = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
ax = ax.flatten()

for col in df.columns:
    sns.boxplot(y=col, data=df, ax=ax[index])
    index += 1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)
plt.show()

# Creating distribution plots
fig, ax = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
ax = ax.flatten()

for col in df.columns:
    sns.histplot(df[col], kde=True, ax=ax[index])
    index += 1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)
plt.show()

# Min-max normalization for selected columns
cols = ['CRIM', 'ZN', 'TAX', 'B']
for col in cols:
    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

# Creating distribution plots again after normalization
fig, ax = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
ax = ax.flatten()

for col in df.columns:
    sns.histplot(df[col], kde=True, ax=ax[index])
    index += 1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)
plt.show()

# Performing standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Fitting and transforming the data
scaled_cols = scaler.fit_transform(df[cols])
scaled_cols = pd.DataFrame(scaled_cols, columns=cols)
df[cols] = scaled_cols

# Creating distribution plots after standardization
fig, ax = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
ax = ax.flatten()

for col in df.columns:
    sns.histplot(df[col], kde=True, ax=ax[index])
    index += 1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)
plt.show()

# Splitting input and target variables
X = df.drop(columns=['MEDV', 'RAD'], axis=1)
y = df['MEDV']

# Function to train and evaluate a model
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error

def train(model, X, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    cv_score = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)
    cv_score = np.abs(np.mean(cv_score))
    
    print("Model Report")
    print("MSE:", mean_squared_error(y_test, pred))
    print('CV Score:', cv_score)

# Removing rows with missing values from the dataset (if any)
X.dropna(axis=0, inplace=True)
y = y[X.index]  # Update the target values accordingly

# Linear Regression Model
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
train(lr_model, X, y)

# Plotting Linear Regression coefficients
coef = pd.Series(lr_model.coef_, X.columns).sort_values()
coef.plot(kind='bar', title='Linear Regression Coefficients')
plt.show()

# Decision Tree Regressor Model
from sklearn.tree import DecisionTreeRegressor
dt_model = DecisionTreeRegressor()
train(dt_model, X, y)

# Plotting Decision Tree feature importance
coef = pd.Series(dt_model.feature_importances_, X.columns).sort_values(ascending=False)
coef.plot(kind='bar', title='Decision Tree Feature Importance')
plt.show()

# Extra Trees Regressor Model
from sklearn.ensemble import ExtraTreesRegressor
et_model = ExtraTreesRegressor()
train(et_model, X, y)

# Plotting Extra Trees feature importance
coef = pd.Series(et_model.feature_importances_, X.columns).sort_values(ascending=False)
coef.plot(kind='bar', title='Extra Trees Feature Importance')
plt.show()

# XGBoost Regressor Model
import xgboost as xgb
xgb_model = xgb.XGBRegressor()
train(xgb_model, X, y)

# Plotting XGBoost feature importance
coef = pd.Series(xgb_model.feature_importances_, X.columns).sort_values(ascending=False)
coef.plot(kind='bar', title='XGBoost Feature Importance')
plt.show()
