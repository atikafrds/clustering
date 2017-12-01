# coding: utf-8
from sklearn.metrics.pairwise import pairwise_distances
import sys
import numpy as np
import kmedoids
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
print("Data shape : ")
print(data.shape)
print(data)

print('Done creating data, start calculating pairwise distances')

import sys
from sklearn.metrics.pairwise import pairwise_distances
D = pairwise_distances(data, metric='euclidean', n_jobs=1)
print("Pairwise shape : ")
print(D.shape)
print('Done creating distance matrix, start calculating pairwise distances')
# np.save('all_data.npy', arr)

# split into 2 clusters
M, C = kmedoids.kMedoids(D, 2)

print('medoids:')
for point_idx in M:
    print( data[point_idx] )

print('')
print('clustering result:')
for label in C:
    for point_idx in C[label]:
        print('label {0}:　{1}'.format(label, data[point_idx]))


# # coding: utf-8
# from sklearn.metrics.pairwise import pairwise_distances
# import sys
# import numpy as np
# import kmedoids
# import pandas as pd
#
# df = pd.read_csv("CencusIncome_data_preprocess.csv")
#
# jumlah = int(sys.argv[1])
# if(jumlah == -1):
#     jumlah = len(df)
#
# arr = []
# for i in range(0, jumlah):
#     arr_temp = []
#     for j in range(0, len(df.iloc[0])):
#         arr_temp.append(df.iloc[i][j])
#     arr.append(arr_temp)
#
# import numpy as np
# data = np.array(arr)
# print('Done creating data, start calculating pairwise distances')
#
# import sys
# from sklearn.metrics.pairwise import pairwise_distances
# D = pairwise_distances(data, metric='euclidean', n_jobs=int(sys.argv[2]))
# np.save(sys.argv[3], arr)
#
# # split into 2 clusters
# M, C = kmedoids.kMedoids(D, 2)
#
# print('medoids:')
# for point_idx in M:
#     print( data[point_idx] )
#
# print('')
# print('clustering result:')
# for label in C:
#     for point_idx in C[label]:
#         print('label {0}:　{1}'.format(label, data[point_idx]))
