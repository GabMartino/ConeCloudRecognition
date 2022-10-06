

from sklearn.cluster import DBSCAN, OPTICS, KMeans
import matplotlib.pyplot as plt
import numpy as np
import copy
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
import csv
def ExtractLabels(path, len_of_Dataset):

    labels = []
    with open(str(path)) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            labels = row

    print(labels)
    labels = [int(i) for i in labels]

    labels_norm = []
    # print(len(DatasetShape))
    # exit()
    for i in range(len_of_Dataset):
        if i in labels:
            labels_norm.append(1)
        else:
            labels_norm.append(0)

    return np.asfarray(labels_norm)

def clusteringDBSCAN(X, expectedSize = 250, min_samples = 5, eps=0.1):


    bdim = [[point[0], point[1]] for point in X]

    db = DBSCAN(eps= eps,  metric='euclidean', min_samples=1).fit(bdim)
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

        # xy = X[class_member_mask ]
        # plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
        #          markeredgecolor='k', markersize=5)

    plt.legend(unique_labels)
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
    #
    #
    # x = []
    # for i in range(0, n_clusters_):
    #     print(len(X[labels == i]))
    #     if len(X[labels == i]) < expectedSize:
    #         x.append(X[labels == i])
    #
    #
    # print(len(x))
    # return x
    return labels


if __name__ == "__main__":


    pcd = []
    start = 0
    for i in range(start, start + 417):
        # if labels[i] == 1:
            # print(i)
            name = "featureExtraction1/double_curve_pushed_1/" + str(i) + ".pcd"
            pcd.append(o3d.io.read_point_cloud(name))






    o3d.visualization.draw_geometries(pcd)

    exit(0)
    centers = []
    for i in range(len(pcd)):
        centers.append(pcd[i].get_center())

    # print(len(centers))
    centers = np.asarray(centers)
    #
    nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(centers)
    distances, indices = nbrs.kneighbors(centers)
    print(distances)
    maxDistances = np.amax(distances, 1)
    print(maxDistances)
    maxDistances.sort()
    plt.scatter(list(range(0, maxDistances.size)), maxDistances)
    plt.title('Scatter plot of K-dist for heuristic Eps Estimation')
    plt.xlabel('K-distances points')
    plt.ylabel('K-distances')
    plt.show()
    print(centers)


    labels = clusteringDBSCAN(centers, eps=0.5)
    labels = np.asarray(labels)

    print(labels)
    # exit(0)
    # newpcd = []
    # j = 0
    # coneIndexes = []
    # # values = [0, 4]
    blue = [1, 4 ]
    yellow = [ 0 , 5]
    noCode = [2, 3, 6]
    labelsDataset = []
    for i in labels:
        if i in blue:
            labelsDataset.append(1)
        elif i in yellow:
            labelsDataset.append(2)
        elif i in noCode:
            labelsDataset.append(0)


    print(labelsDataset)
    # exit(0)

    #
    #     j += 1
    #
    #
    #
    #
    # o3d.visualization.draw_geometries(newpcd)
    # print(coneIndexes)
    # coneIndexes = np.asarray(coneIndexes)
    # coneIndexes = coneIndexes + start
    # print(coneIndexes)
    # coneIndexes = [ i for i in coneIndexes]
    # print(coneIndexes)
    # exit(0)

    with open("featureExtraction1/accelleration_pushed_cones_inverted/labelsCones.csv", "a") as myfile:
        myfile.write(str(labelsDataset))





    # visualizer = o3d.visualization.VisualizerWithEditing()

    # visualizer.create_window()
    #
    # for i in range(0, 100):
    #     visualizer.add_geometry(pcd[i])

    # o3d.visualization.draw_geometries(pcd)
    # v = o3d.visualization.draw_geometries_with_editing(pcd)
    # print(v.get_pickedr_points())
    # visualizer.run()
    #
    # # v = visualizer.get_picked_points()
    # visualizer.destroy_window()
    # # print(v)
    # # o3d.visualization.VisualizerWithEditing(pcd)
    # # print("Inserisci la label per: " + str(i))
    # # label = input()
    # # text = str(label) + ","
    # # with open("testOutput2/labels.csv", "a") as myfile:
    # #     myfile.write(text)