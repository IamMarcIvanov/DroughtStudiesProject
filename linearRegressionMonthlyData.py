import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# loading the raw data
dependent_variables_file_path = r'D:\Mimisbrunnr\Data\Civil_Data\All Combined\inm_monthly_predictors_Point_1.csv'
independent_variable_file_path = r'D:\Mimisbrunnr\Data\Civil_Data\rainfall monthly\Monthly_rain_Point_1.csv'
X_raw = pd.read_csv(dependent_variables_file_path)
y_raw = pd.read_csv(independent_variable_file_path)

# data preprocessing
X_raw.drop(list(range(12)), inplace=True)  # remove year 1950 since it is not in IMD data
X_raw.reset_index(drop=True, inplace=True)  # set indices to start from 0 again
X_raw.drop(columns=['Unnamed: 0', 'lat', 'lon', 'time'], inplace=True)  # drop columns that are not useful
years = [y for y in range(1951, 2015) for _ in range(12)]  # splitting the time into year and month - generate years
X_raw = X_raw.assign(year=years)  # add years
months = []  # generate months
for _ in range(1951, 2015):
    for m in range(1, 13):
        months.append(m)
X_raw = X_raw.assign(month=months)  # add months

y_raw.drop(columns=['LONGITUDE', 'LATITUDE', 'TIME'], inplace=True)  # drop useless columns from imd data
y_raw.drop(list(range(768, 840)), inplace=True)  # drop years from 2015 and onwards

# standardizing the raw data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
y_scaled = scaler.fit_transform(y_raw)

# test train split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, shuffle=False, test_size=0.3)

# performing linear regression
linReg = LinearRegression().fit(X_train, y_train)
print(linReg.score(X_test, y_test))
