import numpy as np
import open3d as o3d
import scipy as sp
from pyntcloud import PyntCloud
from matplotlib import pyplot as plt
import statistics
import pandas as pd
import hdbscan
from pyclustering.cluster.clique import clique
from pyclustering.cluster import cluster_visualizer
from scipy.spatial import distance_matrix
from sklearn.cluster import DBSCAN, OPTICS, KMeans
from sklearn.metrics.pairwise import euclidean_distances
import scipy.optimize as optimize
from sklearn.metrics import r2_score
import pointCloudSupportLibrary as lib
from polylidar import extractPlanesAndPolygons, extractPolygons, Delaunator

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random

from scipy.spatial import distance


def clusteringDBSCAN(X, expectedSize = 250, min_samples = 5, eps=0.1):


    bdim = [[point[0], point[1]] for point in X]

    db = DBSCAN(eps=0.1,  metric='euclidean', min_samples=5).fit(bdim)
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

        xy = X[class_member_mask ]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=5)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()


    x = []
    for i in range(0, n_clusters_):
        print(len(X[labels == i]))
        if len(X[labels == i]) < expectedSize:
            x.append(X[labels == i])


    print(len(x))
    return x

def clusteringOPTICS(X):
    db = OPTICS(eps=0.1,  metric='euclidean', min_samples=10).fit(X)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print(n_clusters_)
    x = []
    for i in range(0, n_clusters_):
        x.append(X[labels == i])

    return x

def clusteringHDBSCAN(X):


    bdim = [[point[0], point[1]] for point in X]
    clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
            gen_min_span_tree=False, leaf_size=40,
            metric='euclidean', min_cluster_size=10, min_samples=None, p=None)
    clusterer.fit(bdim)

    labels = clusterer.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print(n_clusters_)
    #
    # # Black removed and is used for noise instead.
    # unique_labels = set(labels)
    # colors = [plt.cm.Spectral(each)
    #           for each in np.linspace(0, 1, len(unique_labels))]
    # for k, col in zip(unique_labels, colors):
    #     if k == -1:
    #         # Black used for noise.
    #         col = [0, 0, 0, 1]
    #
    #     class_member_mask = (labels == k)
    #
    #     xy = X[class_member_mask]
    #     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
    #              markeredgecolor='k', markersize=16)
    #
    #     xy = X[class_member_mask]
    #     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
    #              markeredgecolor='k', markersize=10)
    #
    # plt.title('Estimated number of clusters: %d' % n_clusters_)
    # plt.show()

    x = []
    for i in range(0, n_clusters_):

        x.append(X[labels == i])

    return x

def clusteringCLIQUE(X, interval = 5, threshold = 10):
    clique_instance = clique(X, amount_intervals = interval,density_threshold = threshold )
    # Run cluster analysis.
    clique_instance.process()
    # Get allocated clusters.
    clusters = clique_instance.get_clusters()
    # Visualize obtained clusters.
    visualizer = cluster_visualizer()
    visualizer.append_clusters(clusters, X)
    visualizer.show()


def findCones(X, coneRadius, minconeRadius = 0.10):
    cones = []
    for cluster in X:
        bidimentionalCluster = [[point[0], point[1]] for point in cluster]
        zValues = [[point[2]] for point in cluster]

        meanHeight = np.mean(zValues)

        # print(type(bidimentionalCluster))
        radius = distance_matrix(np.asarray(bidimentionalCluster), (np.asarray(bidimentionalCluster)))

        if np.max(radius) <= coneRadius**2 and np.max(radius)  >= minconeRadius:
            # print(np.max(radius))
            # print(meanHeight)
            cones.append(cluster)

    return cones


def extractFeatures(maybeCones):
    featuresExtracted = []
    featureOfSingleCone = []
    # print(len(maybeCones))

    for maybeCone in maybeCones:
        matrix = np.asmatrix(maybeCone)
        # print(matrix[:, 0])
        # featureOfSingleCone.append(np.std(np.asarray(matrix[:, 0])))
        # featureOfSingleCone.append(np.std(np.asarray(matrix[:, 1])))
        featureOfSingleCone.append(np.std(np.asarray(matrix[:, 2])))
        featureOfSingleCone.append(np.asarray(max(matrix[:, 2]) - min(matrix[:, 2])))
        # featureOfSingleCone.append(sp.stats.zscore(maybeCone, axis=2))
        # plt.scatter(featureOfSingleCone[:, 0], featureOfSingleCone[:, 1])
        featuresExtracted.append(featureOfSingleCone.copy())
        featureOfSingleCone.clear()


    print(len(featuresExtracted))
    print(featuresExtracted)
    return featuresExtracted


def clusterConesNotCOnes(X):

    kmeans = KMeans(n_clusters=2, random_state=0, init="k-means++").fit(X)

    labels = kmeans.labels_
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
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()

    x = []
    for i in range(0, n_clusters_):
        print(len(X[labels == i]))
        if len(X[labels == i]) < 300:
            x.append(X[labels == i])

    print(len(x))
    return x


def func(data, a, b, c, d):
    return -np.sqrt(abs(c)*(data[:, 0] - a)**2 + abs(c)*(data[:, 1] - b)**2) - d


def func1(X, Y, a, b, c, d):
    return -np.sqrt(abs(c)*(X - a)**2 + abs(c)*(Y - b)**2) - d

#
# def func(data, a, b, c):
#     return c*(data[:, 0] - a)**2 + c*(data[:, 1] - b)**2 - 1

# def predictedCone(X, a, b, c, d):
#
#     ZPredicted = []
#     for point in X:
#         ZPredicted.append(-np.sqrt(abs(c)*(point[0] - a)**2 + abs(c)*(point[1] - b)**2) - d)
#
#     return ZPredicted

def predictedCilinder(X, a, b, c):

    ZPredicted = []
    for point in X:
        ZPredicted.append(abs(c)*(point[0] - a)**2 + abs(c)*(point[1] - b)**2 - 1)

    return ZPredicted

def fittingTest(X):

    guess = (1, 1, 13, 1)
    # params, pcov = []
    # print(X[:, :2])
    params = []
    try:
        params, pcov = optimize.curve_fit(func, X[:, :2], X[:, 2], guess, maxfev=1000000)
    except:
        pass
    return params

def cilinderFiltering(X, cilinderRadius= 0.25, coneMaxHeight = 0.25, coneMinHeight = 0.01 ):
    meanX = np.mean(X[:, 0])
    meanY = np.mean(X[:, 1])
    for point in X:
        if (point[0] - meanX)**2 + (point[1] - meanY)**2 > cilinderRadius**2 :
            return False

    return True

def test(X):


    pcpanda = pd.DataFrame(X[1])
    pcpanda.columns = ["x", "y", "z"]

    cloud = PyntCloud(pcpanda)
    # structures
    kdtree_id = cloud.add_structure("kdtree")
    print(kdtree_id)
    # neighbors
    k_neighbors = cloud.get_neighbors(k=5, kdtree=kdtree_id)
    print(len(X[1]))
    print(len(k_neighbors))
    print(k_neighbors)
    # scalar_fields
    ev = cloud.add_scalar_field("eigen_values", k_neighbors=k_neighbors)
    print(cloud)
    cloud.add_scalar_field("anisotropy", ev= ev)

    cloud.add_scalar_field("curvature", ev=ev)
    cloud.add_scalar_field("eigenentropy", ev=ev)
    cloud.add_scalar_field("eigen_sum", ev=ev)
    cloud.add_scalar_field("linearity", ev=ev)
    # cloud.add_scalar_field("ommnivariance", ev=ev)
    cloud.add_scalar_field("planarity", ev=ev)
    cloud.add_scalar_field("sphericity", ev=ev)
    pcpanda = pd.DataFrame(cloud.points)
    pc2 = pcpanda.drop(["x", "y", "z", "e1(6)", "e2(6)", "e3(6)"], axis=1)


    return pc2.values



if __name__ == "__main__":


    pcd = o3d.io.read_point_cloud("testDinamico/946685593.699955000.pcd")
    points = np.asarray(pcd.points)

    kwargs = dict(alpha=0.0, lmax=1.0)

    # You want everything!
    delaunay, planes, polygons = extractPlanesAndPolygons(points, **kwargs)

    print(polygons)
    # print(points.size)
    o3d.visualization.draw_geometries([pcd])

    firstFiltering = lib.filteruselesspoints(points, 15, 0.5, 30, fun=2)

    # pcd.points = o3d.utility.Vector3dVector(firstFiltering)
    #
    # o3d.visualization.draw_geometries([pcd])
    secondFiltering = lib.filterHigherPoints(firstFiltering, 0.01, 0.1)

    # pcd.points = o3d.utility.Vector3dVector(secondFiltering)
    #
    # o3d.visualization.draw_geometries([pcd])

    groundRemoved = lib.removeGround(secondFiltering, 0.04, 700)
    # pcd.points = o3d.utility.Vector3dVector(groundRemoved)
    # print(groundRemoved.size)
    # o3d.visualization.draw_geometries([pcd], window_name="WITH GROUND REMOVED")
    #
    #
    # pc = np.asarray(groundRemoved)
    #
    # pcpanda = pd.DataFrame(pc)
    # pcpanda.columns = ["x", "y", "z"]
    #
    # cloud = PyntCloud(pcpanda)
    # kdtree_id = cloud.add_structure("kdtree")
    # print(kdtree_id)
    # # neighbors
    # k_neighbors = cloud.get_neighbors(k=5, kdtree=kdtree_id)
    # # print(len(X[1]))
    # print(len(k_neighbors))
    # print(k_neighbors)
    #
    # k_dist = []
    # index = 0
    # for point in pc:
    #     distances = []
    #     for i in range(0, 4):
    #         distances.append(distance.euclidean(point, pc[k_neighbors[index][i]]))
    #
    #     index += 1
    #     k_dist.append(max(distances))
    #     distances.clear()
    #
    #
    # print(k_dist)
    # sorted_kdist = np.sort(k_dist)
    # plt.scatter(range(0, len(sorted_kdist)), sorted_kdist)
    # plt.show()


    # clus = clusteringDBSCAN(np.asarray(groundRemoved))


    clus1 = clusteringHDBSCAN(np.asarray(groundRemoved))
    # index = 0
    # for pcTest in clus1:
    #     print(index)
    #     index +=1
    #     pcd.points = o3d.utility.Vector3dVector(np.vstack(pcTest))
    #     # print(groundRemoved.size)
    #     o3d.visualization.draw_geometries([pcd], window_name="AFTER CLUSTERING")

    onlyCOnes = []
    testCones = []
    coneParam = []
    x = np.linspace(-6, 6, 30)
    y = np.linspace(-6, 6, 30)
    # print(params)
    # print(len(params))
    X, Y = np.meshgrid(x, y)
    for cluster in clus1:
        params = fittingTest(cluster)

        if len(params) > 0 and params[2] > 1*10**-15 and -2 < params[3] < 1:
            print(params)
            onlyCOnes.append(cluster)

            # print(np.column_stack((X, Y)))
            Z = func1(X, Y, 0, 0, params[2], params[3])
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.contour3D(X, Y, Z, 50,  cmap='binary')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set( zlim = (0.1, -1))
            plt.show()


    #
    #
    if len(onlyCOnes) > 1:
        pointCloud = np.vstack(onlyCOnes)
    else:
        pointCloud = onlyCOnes
    # print(len(onlyCOnes))
    #
    pcd.points = o3d.utility.Vector3dVector(pointCloud)
    o3d.visualization.draw_geometries([pcd], window_name="After code finding")

    # clus = clusteringDBSCAN(np.asarray(groundRemoved), expectedSize= 50,  min_samples= 10, eps=0.3)
    # pointCloud = np.vstack(clus)
    # pcd.points = o3d.utility.Vector3dVector(pointCloud)
    # o3d.visualization.draw_geometries([pcd], window_name="after filtering")
