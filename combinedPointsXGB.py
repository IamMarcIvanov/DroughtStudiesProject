import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score


path = r"D:\Mimisbrunnr\Data\Civil_Data\Processed\un_standardised_pt_1.csv"
df = pd.read_csv(path)
dataframes = [pd.read_csv(path)]
for i in range(2, 59):
    path = r"D:\Mimisbrunnr\Data\Civil_Data\Processed\un_standardised_pt_{}.csv".format(
        i
    )
    data = pd.read_csv(path)
    dataframes.append(data)
    df = df.append(data, ignore_index=True)

X_raw = df.drop(columns=["rainfall"])
y_raw = df["rainfall"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
y_scaled = scaler.fit_transform(y_raw.to_numpy().reshape(-1, 1))

# test train split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, shuffle=True, test_size=0.3
)
y_train = y_train.ravel()

xgb = XGBRegressor(max_depth=10, learning_rate=0.2)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
print(r2_score(y_test, y_pred))
