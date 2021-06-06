import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from ray import tune


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
y_train = y_train.ravel()


# performing support vector regression
def getScore(config):
    svr = SVR(C=config['C'], tol=config['tol']).fit(X_train, y_train)
    r2 = svr.score(X_test, y_test)
    tune.report(r2_score=r2)


print('here')
search_space = {"C": tune.uniform(0.1, 2.5),
                'tol': tune.loguniform(1e-6, 1e-2)}
analysis = tune.run(getScore, config=search_space)
dfs = analysis.trial_dataframes
[d.r2_score.plot() for d in dfs.values()]
