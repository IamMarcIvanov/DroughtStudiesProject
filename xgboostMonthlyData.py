import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table


con = Console()

# loading the raw data
dependent_variables_file_path = r'D:\Mimisbrunnr\Data\Civil_Data\All Combined\inm_monthly_predictors_Point_1.csv'
independent_variable_file_path = r'D:\Mimisbrunnr\Data\Civil_Data\rainfall monthly\Monthly_rain_Point_1.csv'
X_raw = pd.read_csv(dependent_variables_file_path)
y_raw = pd.read_csv(independent_variable_file_path)

# data preprocessing
# remove year 1950 since it is not in IMD data
X_raw.drop(list(range(12)), inplace=True)
X_raw.reset_index(drop=True, inplace=True)  # set indices to start from 0 again
X_raw.drop(columns=['Unnamed: 0', 'lat', 'lon', 'time'],
           inplace=True)  # drop columns that are not useful
# splitting the time into year and month - generate years
years = [y for y in range(1951, 2015) for _ in range(12)]
X_raw = X_raw.assign(year=years)  # add years
months = []  # generate months
for _ in range(1951, 2015):
    for m in range(1, 13):
        months.append(m)
X_raw = X_raw.assign(month=months)  # add months

y_raw.drop(columns=['LONGITUDE', 'LATITUDE', 'TIME'],
           inplace=True)  # drop useless columns from imd data
# drop years from 2015 and onwards
y_raw.drop(list(range(768, 840)), inplace=True)

# standardizing the raw data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
y_scaled = scaler.fit_transform(y_raw)

# test train split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, shuffle=False, test_size=0.3)

# create table
table = Table(title='xgb results')
table.add_column('learning rate')
table.add_column('n_estimators')
table.add_column('max_depth')
table.add_column('R2')


# def xgb(max_depth, n_estimators, learning_rate):
#     xgb = XGBRegressor(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate)
#     xgb.fit(X_train, y_train)
#     y_pred = xgb.predict(X_test)
#     r2 = r2_score(y_test, y_pred)
#     return r2


# # using xgboost
# max_r2 = 0
# for lr in np.linspace(0.1, 0.6, 30):
#     for n in range(20, 100):
#         for d in range(1, 15):
#             r2 = xgb(d, n, lr)
#             if r2 > max_r2:
#                 max_r2 = r2
#                 table.add_row(str(lr), str(n), str(d), str(r2))
# con.print(table)

xgb = XGBRegressor(max_depth=2,
                   colsample_bytree=0.0356783,
                   gamma=0.887207,
                   importance_type='total_gain',
                   n_estimators=953,
                   learning_rate=0.280498,
                   reg_alpha=9.56598,
                   reg_lambda=6.99343,
                   subsample=0.288833,)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(r2)
