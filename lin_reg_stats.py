# In this one I will be performing statistical analysis using statsmodels api on INM-CM4-8 point 1 data.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# X is INM data
# y is IMD data
x_path = r'D:\Mimisbrunnr\Data\Civil_Data\INMCM4-8_historical_predictors_Point_1.csv'
y_path = r'D:\Mimisbrunnr\Data\Civil_Data\IMD_Rainfall_0.25x0.25\rain_Point_1.csv'
X = pd.read_csv(x_path)
y = pd.read_csv(y_path)
# dropping longitude and latitude since they are fixed
X.drop(columns=['lon', 'lat', 'nos'], inplace=True)
y.drop(columns=['LONGITUDE', 'LATITUDE'], inplace=True)
# removing duplicates
X.drop_duplicates(keep='first', inplace=True)
# removing everything beyond 2015-01-01 from IMD data
y.drop([i for i in range(23376, len(y))], inplace=True)
# removing everything before 1951-01-01 from INM data
X.drop([i for i in range(365)], inplace=True)
# IMD data has leap years, INM data does not. removing them
y.drop(y[y['TIME'].map(lambda s: s.split()[0][-5:]) == '02-29'].index, inplace = True)
# set indices so that they start from 0
X.reset_index(inplace=True, drop=True)
y.reset_index(inplace=True, drop=True)
# cannot have a string column, so split it into 3
X_dtCol = pd.to_datetime(X['time'], format='%Y-%m-%d %H:%M:%S')
X['time'] = X_dtCol
# convert datetime to day, month and year
# for X
X['day'] = X['time'].dt.day
X['month'] = X['time'].dt.month
X['year'] = X['time'].dt.year
X.drop(columns=['time'], inplace=True)
# for y - commented out since y must have just rainfall
# y_dtCol = pd.to_datetime(y['TIME'], format='%Y-%m-%d %H:%M:%S')
# y['TIME'] = y_dtCol
# y['day'] = y['TIME'].dt.day
# y['month'] = y['TIME'].dt.month
# y['year'] = y['TIME'].dt.year
y.drop(columns=['TIME'], inplace=True)
# print(X.head())
# print(y.head())
y['RAINFALL'].fillna(value=y['RAINFALL'].mean(), inplace=True)

# %%
# Standardising
# if we scale then Condition Number is good in statmodels
scaler = StandardScaler()
X = scaler.fit_transform(X[X.columns])
y = scaler.fit_transform(y)

# %%
# printing
print('X.head()')
print(X.head())
print('y.head()')
print(y.head())
print('X.tail()')
print(X.tail())
print('y.tail()')
print(y.tail())
print()
print('X.columns')
print(X.columns)

# %%
# generating test and train sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# %%
# to compare and check whether X and y have the same set of dates
with open('inm-cm4-8.txt', 'w') as f:
    for ind in range(len(X)):
        f.write(str(X['time'].iloc[ind]).split(' ')[0] + '\n')
with open('imd.txt', 'w') as f:
    for ind in range(len(y)):
        f.write(str(y['TIME'].iloc[ind]).split(' ')[0] + '\n')
# %%
# Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
predictions = np.where(predictions<=0, 0, predictions)
print(f'The R2 Score is: {r2_score(y_test, predictions)}')

# %%
# Polynomial Regression with variable degree
model = make_pipeline(PolynomialFeatures(degree=3),LinearRegression())
model.fit(X_train, y_train)
predictions = model.predict(X_test)
predictions = np.where(predictions<=0, 0, predictions)
print(f'The R2 Score is: {r2_score(y_test, predictions)}')

# %%
# Perform Linear Regression using Ordinary Least Squares and generate a hell of a lot of statistics
# Link: https://medium.com/swlh/interpreting-linear-regression-through-statsmodels-summary-4796d359035a
X = sm.add_constant(X)
model = sm.OLS(y, X)
res = model.fit()
print(res.summary())