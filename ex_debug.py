# coding: utf-8
from sklearn.metrics.pairwise import pairwise_distances
import sys
import numpy as np
import kmedoids
import pandas as pd
import numpy as np

import time
start_time = time.time()

df = pd.read_csv("CencusIncome_data_preprocess.csv")

arr = []
for i in range(0, int(sys.argv[1])):
    arr_temp = []
    for j in range(0, len(df.iloc[0])):
        arr_temp.append(df.iloc[i][j])
    arr.append(arr_temp)
data = np.array(arr)

# data = np.array([
#     [1,2],
#     [3,4],
#     [5,6],
#     [7,8],
#     [9,10]
# ])

print("Data shape : ")
print(data.shape)
print(data)

print('Done creating data, start calculating pairwise distances')

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer

if(len(sys.argv)>4):
    if(sys.argv[4] == '1'):
        scaler = StandardScaler()
    elif(sys.argv[4] == '2'):
        scaler = MinMaxScaler()
    elif(sys.argv[4] == '3'):
        scaler = MaxAbsScaler()
    elif(sys.argv[4] == '4'):
        scaler = RobustScaler()
    elif(sys.argv[4] == '5'):
        scaler = Normalizer()
    data = scaler.fit_transform(data)

print(data[0:5])

from sklearn.metrics.pairwise import pairwise_distances
D = pairwise_distances(data, metric='euclidean', n_jobs=1)
print("Pairwise shape : ")
print(D.shape)
# np.save('all_data.npy', D)
print("Done creating distance matrix, start the algorithm")

# split into 2 clusters
M, C = kmedoids.kMedoids(data, D, int(sys.argv[2]))

st = ''
print('medoids:')
for point_idx in M:
    print( data[point_idx] )
    st = st + str(point_idx) + '\n'
f = open(sys.argv[3], 'w')
f.write(st)

st = ''
print('')
print('clustering result:')
for label in C:
    for point_idx in C[label]:
        print('label {0}:　{1}'.format(label, point_idx))
        st = st + str(label) + " " + str(point_idx) + '\n'
f.write(st)
f.close()

elapsed_time = time.time() - start_time
print("Elapsed time : ")
print(elapsed_time)
