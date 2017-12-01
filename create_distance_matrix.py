import pandas as pd
df = pd.read_csv("CencusIncome_data_preprocess.csv")

arr = []
for i in range(0, len(df)):
    arr_temp = []
    for j in range(0, len(df.iloc[0])):
        arr_temp.append(df.iloc[i][j])
    arr.append(arr_temp)

import numpy as np
data = np.array(arr)
print('Done creating data, start calculating pairwise distances')

import sys
from sklearn.metrics.pairwise import pairwise_distances
D = pairwise_distances(data, metric='euclidean', n_jobs=int(sys.argv[2]))
np.save(sys.argv[1], arr)
