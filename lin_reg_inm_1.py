# In this one I will be performing statistical analysis and linear regression using statsmodels api on INM-CM4-8 point 1 data.

# %%
# imports
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
from sklearn.svm import SVR

# %%
# data preprocessing
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
# X.drop(columns=['time'], inplace=True)
# for y - commented out since y must have just rainfall
# y_dtCol = pd.to_datetime(y['TIME'], format='%Y-%m-%d %H:%M:%S')
# y['TIME'] = y_dtCol
# y['day'] = y['TIME'].dt.day
# y['month'] = y['TIME'].dt.month
# y['year'] = y['TIME'].dt.year
# y.drop(columns=['TIME'], inplace=True)
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
"""
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
"""

# %%
# generating test and train sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# %%
# compare and check whether X and y have the same set of dates
with open('inm-cm4-8.txt', 'w') as f:
    for ind in range(len(X)):
        f.write(str(X['time'].iloc[ind]).split(' ')[0] + '\n')
with open('imd.txt', 'w') as f:
    for ind in range(len(y)):
        f.write(str(y['TIME'].iloc[ind]).split(' ')[0] + '\n')

# %%
# Linear Regression
# Results:
# with and without multioutput='variance_weighted' R2 = 0.020103375252638833, but with standard scaling
# with and without without multioutput='variance_weighted' and without standard scaling -> 0.04127805202148482
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
predictions = np.where(predictions<=0, 0, predictions)
print('The R2 Score is: {}'.format(r2_score(y_test, predictions, multioutput='variance_weighted')))

# %%
# Polynomial Regression with variable degree and no standard scaling
# with and without standard scaling
# for degree = 2 -> 0.05325821822687249
# for degree = 3 -> 0.05630877341597906
# for degree = 4 -> 0.049656805504937784
# for degree = 5 -> 0.02727273524532825
# for degree = 6 -> 0.053098448690447775
# for degree = 7 -> 0.05035674414188229
# for degree = 8 -> 0.03664462816877734

# with and without multioutput='variance_weighted' and scaling done
# for degree = 2 -> 0.029707449653142207
# for degree = 3 -> 0.030248505763899147
# for degree = 4 -> 0.026548069982176847
# for degree = 5 -> 0.005977015889966575
for degree in range(2, 6):
    model = make_pipeline(PolynomialFeatures(degree=degree),LinearRegression())
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    predictions = np.where(predictions<=0, 0, predictions)
    print('The R2 Score is: {}'.format(r2_score(y_test, predictions, multioutput='variance_weighted')))

# %%
# Perform Linear Regression using Ordinary Least Squares and generate a hell of a lot of statistics
# Link: https://medium.com/swlh/interpreting-linear-regression-through-statsmodels-summary-4796d359035a
"""
X = sm.add_constant(X)
model = sm.OLS(y, X)
res = model.fit()
print(res.summary())
"""
# %%
# generate the autocorrelation matrix of the rainfall time series data
# autocorrelation is the correlation of a given sequence with itself as a function of time lag


# %%
# SVR
# Hyperparameters:
# kernel = {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’}, default=’rbf’
# degree: int, default=3 (only for poly)
# gamma{‘scale’, ‘auto’} or float, default=’scale’
#     Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
#     if gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma,
#     if ‘auto’, uses 1 / n_features.
# coef0: float, default=0.0 (only in poly, sigmoid)
# tol: float, default=1e-3
# C: float, default=1.0 (regularisation) always +ve
# epsilon: float, default=0.1
# max_iter: int, default=-1
"""
Results:
kernel      max_iter        degree      gamma       coef0       tol     C       epsilon     R2                          time-taken (s)
linear      150000          -           scale       0.0         1e-3    1.0     0.1         -7.297947472117983e-06
linear      150000          -           auto        0.0         1e-3    1.0     0.1         -7.297947472117983e-06
rbf         150000          -           scale       0.0         1e-3    1.0     0.1         -7.297947472117983e-06
linear      150000          -           auto        0.0         1e-3    0.1     0.1         -7.297947472117983e-06
linear      150000          -           auto        0.0         1e-3    10      0.1         -7.297947472117983e-06
poly        1000000         5           scale       0.0         1e-3    1.0     0.1         -1.839877979259441e-05
poly        1000000\        7           scale       0.0         1e-3    1.0     0.1         -0.0016505121393148858
rbf         -               -           scale       0.0         1e-3    0.001   0.1         -7.297947472117983e-06
rbf         -               -           scale       0.0         1e-3    0.01    0.1         -7.297947472117983e-06
rbf         -               -           scale       0.0         1e-3    0.1     0.1         -7.297947472117983e-06
rbf         -               -           scale       0.0         1e-3    1.0     0.1         -7.297947472117983e-06
rbf         -               -           scale       0.0         1e-3    10.0    0.1         5.346235345804473e-05
rbf         -               -           scale       0.0         1e-3    100     0.1         0.0023060365121116977
rbf         -               -           scale       0.0         1e-3    1000    0.1         0.013924728123650532
rbf         -               -           scale       0.0         1e-3    1000    10          -21.417639281476077         0.6166067123413086
rbf         -               -           scale       0.0         1e-3    1500    10          -16.368526014747765         0.8817434310913086
rbf         -               -           scale       0.0         1e-3    2000    10          -14.06508737925242          1.18355131149292
rbf         -               -           scale       0.0         1e-3    2500    10          -14.045774460318626         1.4827511310577393
rbf         -               -           scale       0.0         1e-3    3000    10          -14.50064409277358          1.785775899887085
rbf         -               -           scale       0.0         1e-3    3000    5           -5.039082037067492          1.8954386711120605
rbf         -               -           scale       0.0         1e-3    5000    5           -4.726379833329805          2.569713830947876
rbf         -               -           scale       0.0         1e-3    7000    5           -4.621419027024011          3.373181104660034
rbf         -               -           scale       0.0         1e-3    9000    5           -4.586154773157784          4.131232500076294
rbf         -               -           scale       0.0         1e-3    1100    5           -4.639946380185331          4.7823326587677
rbf         -               -           scale       0.0         1e-3    1300    5           -4.650582388344436          5.349377393722534
rbf         -               -           scale       0.0         1e-3    1500    5           -4.629035513170665          5.722083806991577
rbf         -               -           scale       0.0         1e-3    1700    5           -4.604327649887012          7.407163143157959
"""

"""
import winsound
from rich_console import console
from rich.table import Table
from tqdm import tqdm, trange
import time

table = Table(title="SVR for INM")
table.add_column('C', style='magenta')
table.add_column('epsilon', style='cyan')
table.add_column('R2', style='green')
table.add_column('time (sec)', style='white')
for c in [3000, 5000, 7000, 9000, 11000, 13000, 15000, 17000]:
    for eps in [5]:
        start_time = time.time()
        console.log('another loop has begun')
        regr = SVR(kernel='rbf', C=c, epsilon=eps)
        console.log('regression model instantiated')
        regr.fit(X_train, np.ravel(y_train))
        console.log('fitting complete')
        predictions = regr.predict(X_test)
        console.log('predictions made')
        predictions = np.where(predictions<=0, 0, predictions)
        console.log('predictions set')
        table.add_row(str(c), str(eps), str(r2_score(y_test, predictions)), str(time.time() - start_time))
        console.log('another iteration complete')
console.print(table)
winsound.Beep(2000, 1000)
"""
# %%
# file logging for SVM hyperparameter optimisation
import winsound
from rich_console import console
from rich.table import Table
from tqdm import tqdm, trange
from rich.live import Live
import time

table = Table(title="SVR for INM")
table.add_column('kernel', style='bright_yellow')
table.add_column('C', style='magenta')
table.add_column('epsilon', style='cyan')
table.add_column('R2', style='green')
table.add_column('time for run (sec)', style='white')

# if below line has file opened as 'w' and file already has useful data, then change open type to 'a'
with open('inm_data_svm_hpp_search.txt', 'a') as f:
    try:
        f.write('All other parameters take their default values.\n')
        f.write('kernel,C,epsilon,r2,time\n')
        with Live(table, refresh_per_second=0.1):
            for kernel in ['rbf',]:
                for c in [1000]:
                    for eps in [0.1]:
                        start_time = time.time()
                        regr = SVR(kernel=kernel, C=c, epsilon=eps)
                        regr.fit(X_train, np.ravel(y_train))
                        predictions = regr.predict(X_test)
                        predictions = np.where(predictions<=0, 0, predictions)
                        time_taken = time.time() - start_time
                        score = r2_score(y_test, predictions)
                        table.add_row(str(kernel), str(c), str(eps), str(score), str(time_taken))
                        f.write(str(kernel) + ',' + str(c) + ',' + str(eps) + ',' + str(score) + ',' + str(time_taken) + '\n')
    except Exception as e:
        print('Error occured')
winsound.Beep(2000, 1000)