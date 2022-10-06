


import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN, OPTICS, KMeans

from sklearn.neighbors import NearestNeighbors


def plotKNNToFindEPS(Dataset):


    nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(Dataset)
    distances, indices = nbrs.kneighbors(Dataset)
    # print(distances)
    maxDistances = np.amax(distances, 1)
    # print(maxDistances)
    maxDistances.sort()
    plt.scatter(list(range(0, maxDistances.size)), maxDistances)
    plt.title('Scatter plot of K-dist for heuristic Eps Estimation')
    plt.xlabel('K-distances points')
    plt.ylabel('K-distances')
    plt.show()


def clusteringDBSCAN(X , EstimatedEps, maxClustersize):


    bdim = [[point[0], point[1]] for point in X]
    db = DBSCAN(EstimatedEps, metric='euclidean', min_samples=15).fit(bdim)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print(n_clusters_)

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=16)

        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=5)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    print(all(v == 0 for v in labels))
    #
    # print(labels[k == -1])
    x = []
    for i in range(0, n_clusters_):
        x.append(X[labels == i])

    clusters = []
    for cluster in x:
        if(len(cluster) <= maxClustersize):
            clusters.append(cluster)

    return  clusters

import  csv
#
# labels = []
#
# for i in range(109, 300 ):
#
#     name = "testOutput2/" + str(i) + ".pcd"
#     pcd = o3d.io.read_point_cloud(name)
#
#     o3d.visualization.draw_geometries([pcd])
#     print("Inserisci la label per: " + str(i))
#     label = input()
#     text = str(label) + ","
#     with open("testOutput2/labels.csv", "a") as myfile:
#         myfile.write(text)


# pcd = []
#
# labels = []
# with open("testOutput2/labels1.csv") as csvfile:
#     readCSV = csv.reader(csvfile, delimiter=',')
#     for row in readCSV:
#         labels = row
#
# print(labels)
# labels = [int(i) for i in labels]
# for i in labels:
#     print(i)
i = 250
name = "test5/" + str(i) + ".pcd"
pcd = o3d.io.read_point_cloud(name)


points = np.asarray(pcd.points)
# plotKNNToFindEPS(points)
clusters = clusteringDBSCAN(points, 0.1, 250)
clustersStacked = np.vstack(clusters)
print(len(clusters))
pcd.points = o3d.utility.Vector3dVector(clustersStacked)
# pcd.points = np.asfarray(clusters[0])
# print()
# print(len(x))
o3d.visualization.draw_geometries([pcd])
# o3d.visualization.draw_geometries(pcd)
