import numpy as np
import pandas as pd
from random import shuffle
from sklearn.decomposition import PCA

path1 = r"E:\BITS\Yr 3 Sem 2\CE F376 Civil Climate change SOP\Drought Studies\Data\Daily Hyderabad_updated.csv"
path2 = r"E:\BITS\Yr 3 Sem 2\CE F376 Civil Climate change SOP\Drought Studies\Data\Daily Hyderabad_interp_updated.csv"

data1 = pd.read_csv(path1)
# print(data1.head())

data1_no_index = data1.drop(columns=["index"], axis=1)
print(data1_no_index.head())
print()
for i in range(10):
    a = data1_no_index.columns.tolist()
    shuffle(a)
    data1_no_index = data1_no_index[a]
    print(data1_no_index.head())
    print()

    pca = PCA(n_components=len(data1_no_index.columns))
    pca.fit(data1_no_index)
    for name, val in zip(data1_no_index.columns, pca.explained_variance_ratio_):
        print("{:<10} {:<30}".format(name, val))
    print()
