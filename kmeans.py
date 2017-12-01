import pandas as pd
import csv
import sys
import numpy as np
import math
import random
import copy
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer

k = 0
max_iter = 0
temp_dataset = None
dataset = None
labels = []
new_label = ""
init_centroids = None
centroids = []
last_centroids = []
data_clusters = []

def euclidian_distance(vector1, vector2):
    if len(vector1) != len(vector2):
        raise Exception("ERROR: can not calculate distance between two points with different lengths")

    distance = 0.0

    for i in range(0, len(vector1)):
        dif = float(vector1[i]) - float(vector2[i])
        distance += (dif ** 2)

    return math.sqrt(distance)


def read_dataset():
    global k
    global temp_dataset
    global dataset
    global max_iter
    global init_centroids

    if (len(sys.argv) != 4) and (len(sys.argv) != 5):
        print("Error: incorrect number of specified parameters.")
    else:
        k = int(sys.argv[1])
        max_iter = int(sys.argv[2])
        temp_dataset = np.genfromtxt(sys.argv[3], delimiter=', ', dtype=None, usecols=(0, 2, 4, 10, 11, 12),
            names=('age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'))
        dataset = [list(elem) for elem in temp_dataset]

        with open(sys.argv[3]) as f:
            for line in f:
                new_label = str(line.split(', ')[-1])
                if (str(new_label) == ">50K\n"):
                    labels.append(0)
                else:
                    labels.append(1)
        if len(sys.argv) == 5:
            init_centroids = np.genfromtxt(sys.argv[4], delimiter=', ', dtype=None)

def generate_random_centroids():
    for k_number in range(k):
        centroid = random.randint(0, len(dataset)-1)
        last_centroids.append(dataset[centroid])
        centroids.append(dataset[centroid])

def initialize_centroids():
    if init_centroids is None:
        generate_random_centroids()
    else:
        for row in range(len(init_centroids)):
            last_centroids.append(init_centroids[row].tolist())
            centroids.append(init_centroids[row].tolist())
        print("panjangan centroids : ",len(centroids[0]))

def assign_clusters():
    global data_clusters

    new_data_clusters = []
    for i in range(len(dataset)):
        centroids_distance = []
        for k_idx in range(k):
            distance = euclidian_distance(dataset[i], centroids[k_idx])
            centroids_distance.append(distance)
        value, idx = min((value, idx) for (idx, value) in enumerate(centroids_distance))
        new_data_clusters.append(idx);

    data_clusters = new_data_clusters

# def purity():
# 	for i in range(k):

def get_new_centroids():
    global last_centroids
    global centroids

    cluster_mean = []

    for cluster_number in range(k):
        summed_atr = [0] * int(len(dataset[0]))

        data_in_cluster_counter = 0.0
        for assignment_row in range(len(data_clusters)):
            if cluster_number == data_clusters[assignment_row]:
                try:
                    summed_atr = [x + y for x, y in zip(summed_atr, dataset[assignment_row])]
                except:
                    print(dataset[assignment_row])
                data_in_cluster_counter += 1.0

        if data_in_cluster_counter == 0.0:
            cluster_mean.append(centroids[cluster_number])
        else:
            mean_for_cluster = [x / data_in_cluster_counter for x in summed_atr]
            cluster_mean.append(mean_for_cluster)

    last_centroids = copy.copy(centroids)
    centroids = copy.copy(cluster_mean)

def has_converged():
    total_distance_moved = 0

    for index in range(len(centroids)):
        distance = euclidian_distance(last_centroids[index], centroids[index])
        total_distance_moved += distance

    if total_distance_moved == 0:
        return True
    else:
        return False

def run_k_means():
    for iteration in range(max_iter):
        if iteration % 10 == 0:
            print("Iteration: ", iteration)

        assign_clusters()

        get_new_centroids()

        if has_converged():
            print("Stop at number of iteration: ", iteration)
            return None

def compare_data():
    counter=0
    if data_clusters.count(0) >= data_clusters.count(1):
        for i in range(len(dataset)):
            if(data_clusters[i]==labels[i]):
                counter+=1
    else:
        for i in range(len(dataset)):
            if(data_clusters[i]!=labels[i]):
                counter+=1
    return counter

def normalize_data():
    global dataset

    # scaler = MaxAbsScaler()
    # scaler = RobustScaler()
    # scaler = Normalizer()
    scaler = MinMaxScaler()
    
    dataset = scaler.fit_transform(dataset)

def run():
    random.seed()
    read_dataset()
    normalize_data(3)
    initialize_centroids()
    print()
    print("INITIAL CENTROIDS")
    for i in range(len(centroids)):
        print("Centroid cluster ", i, ": ", centroids[i])
    print()
    print("Start clustering..")
    print()
    run_k_means()
    print()
    print("CLUSTERING DONE")
    for i in range(len(centroids)):
        print("Centroid Cluster ", i, ": ", centroids[i])
    for i in range(len(centroids)):
        print("Amount of data in cluster ", i, ": ", data_clusters.count(i))
    if (k==2):
    	print()
    	error = round((compare_data()/int(len(dataset)))*100,2)
    	print("Accuracy: ", (100-error), "%")
    	print("Error: ", error, "%")
    # print("\n\n\n")
    # for i in range(len(dataset)):
    #     print(str(i) + " -> Cluster " + str(data_clusters[i]))

run()
