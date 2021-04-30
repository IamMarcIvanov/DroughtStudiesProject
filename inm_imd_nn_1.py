# In this one I will be performing neural network analysis on INM-CM4-8 point 1 data with IMD data for point 1.

# %%
# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from rich_console import console
from rich.table import Table
from tqdm import tqdm, trange
import time
from rich.traceback import install
install()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
X.drop(columns=['time'], inplace=True)
# for y - not done since y must have just rainfall
y.drop(columns=['TIME'], inplace=True)
# fill nan values from IMD data. there are no nan values in INM-CM4-8 data
y['RAINFALL'].fillna(value=y['RAINFALL'].mean(), inplace=True)

# %%
# Standardising
# if we scale then Condition Number is good in statmodels
scaler = StandardScaler()
X = scaler.fit_transform(X[X.columns])
y = scaler.fit_transform(y)

# %%
# generating test and train sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
xTest = torch.from_numpy(X_test).to(device).float()

# %%
# neural network
class NN(nn.Module):
    def __init__(self, input_size, output_size, n_hidden_layers, n_units_hidden_layers):
        super(NN, self).__init__()
        
        self.n_hidden_layers = n_hidden_layers
        self.n_units_hidden_layers = n_units_hidden_layers
        
        def linBlock(inDim, outDim):
            return nn.Sequential(
                nn.Linear(inDim, outDim),
                nn.BatchNorm1d(outDim),
                nn.LeakyReLU())
        
        self.in_layer = linBlock(input_size, self.n_units_hidden_layers)
        self.hidden_layers = nn.ModuleList([linBlock(self.n_units_hidden_layers, self.n_units_hidden_layers)])
        self.out_layer = nn.Sequential(
            nn.Linear(self.n_units_hidden_layers, output_size),
            nn.BatchNorm1d(output_size),
            nn.Sigmoid())
    
    def forward(self, x):
        out = self.in_layer(x)
        for layer in self.hidden_layers:
            out = layer(out)
        out = self.out_layer(out)
        return out

# %%
# torch setup
learning_rate = 0.001
num_epochs = 5000
batch_size = 100000
n_hidden_layers = 6
n_units_hidden_layers = 200

model = NN(X_train.shape[1], 1, n_hidden_layers, n_units_hidden_layers).to(device)
criterion = nn.SmoothL1Loss()
model = model.float()
criterion = criterion.float()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# %%
# local rich setup
table = Table(title='NN for INM data')
table.add_column('iteration number', style='cyan')
table.add_column('loss', style='green')
table.add_column('time taken (sec)', style='magenta')
table.add_column('r2 score', style='bright_yellow')
table.add_column('number of hidden layers', style='white')
table.add_column('number of units per hidden layer', style='white')
table.add_column('number of epochs', style='white')
table.add_column('batch size', style='white')
table.add_column('learning rate', style='white')

# %%
# actual iterations
losses = []
for epoch in trange(num_epochs):
    start_time = time.time()
    
    indices = np.random.choice(len(X_train), batch_size)
    xTrain = torch.from_numpy(X_train[indices]).to(device).float()
    yTrain = torch.from_numpy(y_train[indices]).to(device).float()
    
    outputs = model(xTrain.float())
    loss = criterion(outputs, yTrain.float())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    end_time = time.time()
    if epoch % 1000 == 0:
        with torch.no_grad():
            losses.append(loss.item())
            predictions = model(xTest).to('cpu').numpy()
            table.add_row(str(epoch * batch_size), 
                        str(loss.item()), 
                        str(end_time - start_time),
                        str(r2_score(y_test, predictions, multioutput='variance_weighted')),
                        str(n_hidden_layers),
                        str(n_units_hidden_layers),
                        str(num_epochs),
                        str(batch_size),
                        str(learning_rate))
console.log(table)
predictions = model(xTest).to('cpu').detach().numpy()
console.log('final r2 score: ', r2_score(y_test, predictions, multioutput='variance_weighted'))

# %%
# plot the loss
plt.plot(list(range(len(losses))), losses)
plt.xlabel('iterations in 10^9')
plt.ylabel('loss')
plt.show()