from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import random

def kMedoids(data, D, k, tmax=100):
    # determine dimensions of distance matrix D
    m, n = D.shape

    if k > n:
        raise Exception('too many medoids')
    # randomly initialize an array of k medoid indices
    M = np.arange(n)
    np.random.shuffle(M)
    M = np.sort(M[:k])

    # create a copy of the array of medoid indices
    Mnew = np.copy(M)

    # initialize a dictionary to represent clusters
    C = {}
    for t in range(0, tmax):
        # determine clusters, i. e. arrays of data indices
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
        # update cluster medoids
        for kappa in range(k):
            # data_temp = []
            # for l in range(0, len(C[kappa])):
            #     data_temp.append(data[C[kappa][l]])
            # pairwise = pairwise_distances(np.array(data_temp), metric='euclidean', n_jobs=1)
            # print("Pairwise : ")
            # print(pairwise)

            # J = []
            # for l in C[kappa]:
            #     sum_distance = 0.0
            #     for m in C[kappa]:
            #         sum_distance = sum_distance + D[l][m]
            #         # print(sum_distance)
            #         # print("Sum distance : ")
            #         # print(m)
            #     J.append(float(sum_distance)/float(len(C[kappa])))
            # print("J2 : ")
            # print(J)

            J = []
            x = 10
            increment = int(len(C[kappa])/x)
            for i in range(0, x-1):
                J1 = D[np.ix_(C[kappa][i*increment:(i+1)*increment],C[kappa])]
                # J2 = D[np.ix_(C[kappa][increment:2*increment],C[kappa])]
                # J3 = D[np.ix_(C[kappa][2*increment:3*increment],C[kappa])]
                # J4 = D[np.ix_(C[kappa][3*increment:len(C[kappa])],C[kappa])]
                J1 = np.mean(J1, axis=1)
                # J2 = np.mean(J2, axis=1)
                # J3 = np.mean(J3, axis=1)
                # J4 = np.mean(J4, axis=1)
                J = np.concatenate([J, J1])
            J1 = D[np.ix_(C[kappa][(x-1)*increment:len(C[kappa])], C[kappa])]
            J1 = np.mean(J1, axis=1)
            # print('J')
            # print(J)
            # print('J1')
            # print(J1)
            J = np.concatenate([J, J1])
            # J = np.concatenate([J1, J2, J3, J4])
            # print("J-awal : ")
            # print(J)

            # if np.array_equal(pairwise, J):
            #     print('YESSSS')

            # J = np.mean(J, axis=1)

            # J = np.mean(D[np.ix_(C[kappa],C[kappa])],axis=1)
            j = np.argmin(J)

            # import random
            # x = random.randint(0, len(C[kappa])-1)
            Mnew[kappa] = C[kappa][j]
        np.sort(Mnew)
        # check for convergence
        if np.array_equal(M, Mnew):
            break
        M = np.copy(Mnew)
        print("Iteration : ")
        print(t)
    else:
        # final update of cluster memberships
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
        print("Iteration : ")
        print(t)

    # return results
    return M, C
