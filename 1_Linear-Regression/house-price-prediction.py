import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error

df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_test_copy = pd.read_csv('test.csv')

df = df[['YearBuilt', 'LotArea', 'GrLivArea', 'TotalBsmtSF', 'FullBath' ,'OverallCond', 'OverallQual', 'SalePrice']]
df_test = df_test[['YearBuilt', 'LotArea', 'GrLivArea', 'TotalBsmtSF', 'FullBath' ,'OverallCond', 'OverallQual']]
df_test_copy = df_test_copy[['Id']]

X = df[['YearBuilt', 'LotArea', 'GrLivArea', 'TotalBsmtSF', 'FullBath', 'OverallCond', 'OverallQual']]
y = df['SalePrice']

# Splitting the data into training and testing data
X_test = df_test[['YearBuilt', 'LotArea', 'GrLivArea', 'TotalBsmtSF', 'FullBath', 'OverallCond', 'OverallQual']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Handling missing values
X_train_scaled = (X_train - X_train.mean()) / X_train.std()
X_test_scaled = (X_test - X_train.mean()) / X_train.std()

# Drop rows with missing values
X_train_scaled.dropna(inplace=True)
y_train.dropna(inplace=True)
X_test_scaled.dropna(inplace=True)
y_test.dropna(inplace=True)

# Training the linear regression model
regr = LinearRegression()
regr.fit(X_train_scaled, y_train)

y_pred = regr.predict(X_test_scaled)

df_results = pd.concat([df_test_copy, pd.DataFrame({'SalePrice': y_pred})], axis = 1)
df_results.to_csv('results.csv', index=False)

mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
rmse = mean_squared_error(y_true=y_test, y_pred=y_pred, squared=False)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)