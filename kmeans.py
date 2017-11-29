import pandas as pd
import csv
import sys
import numpy as np
import math
import random
import copy


k = 0
max_iter = 0
dataset = None
labels = []
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
    global dataset
    global max_iter
    global init_centroids

    if (len(sys.argv) != 4) and (len(sys.argv) != 5):
        print("Error: incorrect number of specified parameters.")
    else:
        k = int(sys.argv[1])
        max_iter = int(sys.argv[2])
        dataset = np.genfromtxt(sys.argv[3], delimiter=', ', dtype=None, usecols=(0, 2, 4, 10, 11, 12),
            names=('age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'))
        # labels = np.loadtxt(sys.argv[3], delimiter=',', dtype=str, usecols=(14))
        # labels = np.genfromtxt(sys.argv[3], delimiter=', ', dtype=None, usecols=(14),
        #     names=('label'))
        with open(sys.argv[3]) as f:
            for line in f:
                labels.append(line.split(',')[-1])
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


def assign_clusters():
    global data_clusters

    new_data_clusters = []
    for i in range(len(dataset)):
        distance0 = euclidian_distance(dataset[i], centroids[0])
        distance1 = euclidian_distance(dataset[i], centroids[1])
        if distance0 < distance1:
            new_data_clusters.append(0)
        else:
            new_data_clusters.append(1)

    data_clusters = new_data_clusters


def get_new_centroids():
    global last_centroids
    global centroids

    # List of lists that holds [[coordinates]] for each of the k clusters (averaged per cluster)
    cluster_mean = []

    for cluster_number in range(k):
        summed_atr = [0] * 6

        data_in_cluster_counter = 0.0
        for assignment_row in range(len(data_clusters)):
            if cluster_number == data_clusters[assignment_row]:
                summed_atr = [x + y for x, y in zip(summed_atr, dataset[assignment_row])]
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
        print(iteration)

        assign_clusters()

        get_new_centroids()

        if has_converged():
            return None

def run():
    random.seed()

    read_dataset()
    initialize_centroids()
    print(centroids)
    print(str(len(labels)))
    # print(labels)
    # print(dataset)
    run_k_means()
    print("CENTROIDS\n")
    print(centroids)
    print("Count 0: " + str(data_clusters.count(0)))
    print("Count 1: " + str(data_clusters.count(1)))
    # print("\n\n\n")
    # for i in range(len(dataset)):
    #     print(str(i) + " -> Cluster " + str(data_clusters[i]))

run()
