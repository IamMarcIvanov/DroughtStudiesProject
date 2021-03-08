import numpy as np
import pandas as pd
from random import shuffle
from sklearn.decomposition import PCA
from sklearn import preprocessing

path1 = r"E:\BITS\Yr 3 Sem 2\CE F376 Civil Climate change SOP\Drought Studies\Data\Daily Hyderabad_updated.csv"
path2 = r"E:\BITS\Yr 3 Sem 2\CE F376 Civil Climate change SOP\Drought Studies\Data\Daily Hyderabad_interp_updated.csv"

data1 = pd.read_csv(path1)
# print(data1.head())

data1_no_index = data1.drop(columns=["index"], axis=1)
scaler = preprocessing.StandardScaler().fit(data1_no_index)
data1_no_index_std = pd.DataFrame(scaler.transform(data1_no_index), columns = data1_no_index.columns.tolist())
print(data1_no_index_std.head())
print()
for i in range(10):
    a = data1_no_index_std.columns.tolist()
    shuffle(a)
    data1_no_index_std = data1_no_index_std[a]
    print(data1_no_index_std.head())
    print()

    pca = PCA(n_components=len(data1_no_index_std.columns))
    pca.fit(data1_no_index_std)
    for name, val in zip(data1_no_index_std.columns, pca.explained_variance_ratio_):
        print("{:<10} {:<30}".format(name, val))
    print()
