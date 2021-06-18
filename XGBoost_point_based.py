import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import pickle


class XGBoost:
    def __init__(self, points: list[int], folder: str):
        self.points = points
        self.folder = folder
        self.scaler = StandardScaler()
       
    def load_data(self):
        # data = pd.DataFrame(columns=['clt', 'hurs', 'height', 'huss' , 'tas',uas,vas,year,month,psl,rainfall])
        data = pd.DataFrame()
        for point in self.points:
            data = data.append(pd.read_csv(self.folder + '\\' + 'un_standardised_pt_{}.csv'.format(point)), ignore_index=True)
        self.X = data.drop(columns=['rainfall'])
        self.y = data['rainfall']
        
        
    def standardise_data(self):
        self.X_scaled = self.scaler.fit_transform(self.X)
        self.y_scaled = self.scaler.fit_transform(self.y.to_numpy().reshape(-1, 1))
        
    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_scaled, self.y_scaled, test_size=0.3)        
        
    def training(self):
        self.xgboost = XGBRegressor(learning_rate=0.2, max_depth=10)
        self.xgboost.fit(self.X_train, self.y_train)
        y_pred = self.xgboost.predict(self.X_test)
        return r2_score(self.y_test, y_pred)
    
    def save_model(self, save_path: str):
        with open(save_path, 'wb') as f:
            pickle.dump(self.xgboost, f)
        
        

separate = list(range(1, 9)) + [13, 14]     
xgb1 = XGBoost(points = [i for i in list(range(1, 59)) if i not in separate], folder = r'D:\Svalbard\Data\CivilData\ProcessedData')
xgb1.load_data()
xgb1.standardise_data()
xgb1.split_data()
print(xgb1.training())
xgb1.save_model(r'D:\LibraryOfBabel\Projects\Civil\majority_points_model')

xgb2 = XGBoost(points = separate, folder = r'D:\Svalbard\Data\CivilData\ProcessedData')
xgb2.load_data()
xgb2.standardise_data()
xgb2.split_data()
print(xgb2.training())
xgb2.save_model(r'D:\LibraryOfBabel\Projects\Civil\minority_points_model')
