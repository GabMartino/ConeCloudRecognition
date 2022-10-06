import csv
import numpy as np
import open3d as o3d
import sklearn
from random import  sample
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, OPTICS, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing, svm
import pylab
from sklearn.svm import OneClassSVM
import hdbscan
import matplotlib
import skfuzzy as fuzz

from sklearn import metrics
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_score
import time
from sklearn.feature_selection import RFECV
def clusteringHDBSCAN(X):


    clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
            gen_min_span_tree=False,
            metric='euclidean', min_cluster_size=10, min_samples=None, p=None)
    clusterer.fit(X)

    labels = clusterer.labels_
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
                 markeredgecolor='k', markersize=10)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    x = []
    for i in range(0, n_clusters_):

        x.append(X[labels == i])

    return x
def clusteringDBSCAN(X , EstimatedEps):
    db = DBSCAN(EstimatedEps, metric='euclidean', min_samples=4).fit(X)
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


    print(len(x))
    # return x
    return labels, x



def clearOfOutliers(Dataset, labels):


    indeces = []
    for i in range(len(labels)):
        if(labels[i] == 0):
            indeces.append(i)

    Dataset = np.delete(Dataset, indeces, 0)

    return Dataset




def takeDatasetFromCSV(path):

    Dataset = []
    with open(str(path)) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            Dataset.append(np.asfarray(row, float))

    ##Delete the last row 'cause it may be corrupted
    Dataset.pop(len(Dataset) - 1)
    Dataset = np.vstack(Dataset)
    # print(np.asmatrix(Dataset))

    #delete the first column cause it's the ID
    Dataset = np.delete(Dataset, 0, 1)

    return Dataset

def ExtractPCAFeatures(Dataset, number_of_components = 10):

    pca = PCA(n_components=number_of_components, svd_solver='auto')
    pca.fit(np.nan_to_num(Dataset))
    print("The best component")
    print( pca.explained_variance_ratio_)
    # print(pca.singular_values_)
    DatasetShape = pca.transform(np.nan_to_num(Dataset))

    return DatasetShape, pca

def ExtractLabels(path, len_of_Dataset, areIndexesOfPositiveClass=True):

    labels = []
    with open(str(path)) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            labels = row

    print(labels)
    if areIndexesOfPositiveClass:
        labels = [int(i) for i in labels]

        labels_norm = []
        # print(len(DatasetShape))
        # exit()
        for i in range(len_of_Dataset):
            if i in labels:
                labels_norm.append(1)
            else:
                labels_norm.append(0)

        labels = np.asfarray(labels_norm)
    else:
        labels = np.asfarray(labels)



    return labels


def plotDataset(Dataset, labels):


    colors = ['blue', 'yellow']
    if len(labels) != len(Dataset):
        plt.scatter(Dataset[:, 0], Dataset[:, 1])
    else:
        plt.scatter(Dataset[:, 0], Dataset[:, 1], c=labels, cmap=matplotlib.colors.ListedColormap(colors))

    plt.title('Scatter plot ')
    plt.xlabel('First Component')
    plt.ylabel('Second Component')
    plt.show()

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



def shapeElab():


    DatasetShape = takeDatasetFromCSV("testOutput2/shapeFeatureDatasetPushedCurve1.csv")

    #Normalize
    DatasetShape = preprocessing.scale(DatasetShape, axis=0)
    # DatasetShape = sklearn.preprocessing.normalize(DatasetShape, axis=0)

    DatasetShape = ExtractPCAFeatures(DatasetShape, number_of_components=25)

    labels = ExtractLabels("testOutput2/labels1.csv", len(DatasetShape))

    plotDataset(DatasetShape, labels)


    classifier(DatasetShape, labels)


def intensityElab():

    Dataset = takeDatasetFromCSV("testOutput2/intensityFeatureDatasetPushedCurve1.csv")
    labels = ExtractLabels("testOutput2/labels1.csv", len(Dataset))

    # plotDataset(Dataset, labels)

    DatasetClean = clearOfOutliers(Dataset, labels)
    #
    # plt.scatter(DatasetClean[0:100, 0], DatasetClean[0:100, 1])
    # plt.title('Scatter plot INTENSITY after')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.show()


    Dataset1 = takeDatasetFromCSV("testOutput3/intensityFeatureDatasetAcceleration1.csv")

    labels1 = ExtractLabels("testOutput3/labelsColors.csv", len(Dataset1), areIndexesOfPositiveClass= False)
    # print(labels1)
    # exit(0)
    DatasetTotal = np.vstack((DatasetClean, Dataset1))

    DatasetTotal = preprocessing.scale(DatasetTotal, axis=0)

    DatasetTransformed, pca = ExtractPCAFeatures(DatasetTotal, 5)

    labels1 = np.asarray(labels1)

    labels = np.repeat(2, len(DatasetClean))
    labels = np.asfarray(labels)
    print(labels)
    print(labels1)
    labels = np.concatenate((labels, labels1))
    # labels = np.asarray(labels)
    # labels = labels.tolist()
    # labels.append(labels1)
    # labels = np.vstack((labels, labels1))

    # labels = np.asarray(labels)
    print(labels)
    print(len(labels))
    print(len(DatasetTransformed))
    colors = ['blue', 'yellow', 'black']

    if len(labels) != len(DatasetTransformed):
        plt.scatter(DatasetTransformed[:, 0], DatasetTransformed[:, 1])
    else:
        plt.scatter(DatasetTransformed[:, 0], DatasetTransformed[:, 1], c=labels, cmap=matplotlib.colors.ListedColormap(colors))

    plt.title('Scatter plot ')
    plt.xlabel('First Component')
    plt.ylabel('Second Component')
    plt.show()

    # Dataset1 = pca.transform(Dataset1)

    # plotDataset(Dataset1, labels1)


    # Dataset = preprocessing.normalize(Dataset, axis=0)
    # Dataset = preprocessing.scale(Dataset, axis=0)


    # plotDataset(DatasetTransformed, labels)


    # classifier(DatasetTransformed, labels)


def totalProcessing():


    DatasetShape = takeDatasetFromCSV("testOutput2/shapeFeatureDatasetPushedCurve1.csv")

    labels = ExtractLabels("testOutput2/labels1.csv", len(DatasetShape))

    DatasetIntesity = takeDatasetFromCSV("testOutput2/intensityFeatureDatasetPushedCurve1.csv")

    Dataset = np.hstack((DatasetShape, DatasetIntesity))

    # plotDataset(Dataset, labels)
    Dataset = preprocessing.scale(Dataset, axis=0)

    DatasetTransformed = ExtractPCAFeatures(Dataset, 40)

    # plotDataset(DatasetTransformed, labels)


    classifier(DatasetTransformed, labels)

def classifier(Dataset, y):
    #
    # X_train, X_test, y_train, y_test = train_test_split(Dataset, y, test_size=0.2, random_state= 0)
    # clf = KNeighborsClassifier(n_neighbors=4)
    # model = clf.fit(X_train, y_train)
    #
    # scores = cross_val_score(clf, Dataset, y, cv=10)
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    # # print(scores.mean())
    # # print(model.score(X_test, y_test))
    #
    # clf = RandomForestClassifier(max_depth=10, random_state=0)
    # model = clf.fit(X_train, y_train)
    #
    # scores = cross_val_score(clf, Dataset, y, cv=10)
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    # # print(model.score(X_test, y_test))
    #
    # clf = DecisionTreeClassifier(max_depth=5)
    #
    # scores = cross_val_score(clf, Dataset, y, cv=10)
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    names = ["Linear SVM", "RBF SVM",
             "Decision Tree", "Random Forest", "AdaBoost"]

    classifiers = [
        # KNeighborsClassifier(3),
        # SVC(kernel="linear", C=0.025),
        # SVC(gamma=2, C=1),
        DecisionTreeClassifier(max_depth=10),
        RandomForestClassifier(max_depth=10, n_estimators=10, max_features=1),
        # MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(n_estimators=50)
        # GaussianNB(),
        # QuadraticDiscriminantAnalysis()
        ]

    index = 0
    for clf in classifiers:

        rfecv = RFECV(estimator=clf, step=1, cv=10,
                      scoring='accuracy')
        rfecv.fit(Dataset, y)

        print("Optimal number of features : %d" % rfecv.n_features_)

        # Plot number of features VS. cross-validation scores
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        plt.show()

        #
        # rfe = RFE(estimator=clf, n_features_to_select=10, step=1)
        # selector = rfe.fit(Dataset, y)
        # index1 = 0
        # columnsToDelete = []
        # for i in selector.support_:
        #     if i == False:
        #         columnsToDelete.append(index1)
        #
        #     index1 += 1
        #
        # Dataset = np.delete(Dataset, np.asarray(columnsToDelete), 1)
        # # print(selector.support_)
        # # print(len(Dataset))
        # # Dataset = np.delete(Dataset, [selector.support_[selector.support_ == False]], 1)
        # # print(len(Dataset))
        # # t0 = time.time()
        Dataset = rfecv.transform(Dataset)
        scores = cross_val_score(clf, Dataset, y, cv=10)
        # # print("training time:", round(time.time() - t0, 3), "s"  )# the time would be round to 3 decimal in seconds
        print(names[index])
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 1.96/np.sqrt(10)))
        # index += 1




    # print(model.score(X_test, y_test))
    #
    # print(len(Dataset))
    # DatasetNormal = [Dataset[i] for i, val in enumerate(y) if val == 1]
    # print(len(DatasetNormal))
    # DatasetOutliers = [Dataset[i] for i, val in enumerate(y) if val == 0]
    # print(len(DatasetOutliers))
    # y_normal = np.repeat(1, len(DatasetNormal))
    # y_abnormal = np.repeat(1, len(DatasetOutliers))
    #
    # X_train, X_test, y_train, y_test = train_test_split(DatasetNormal, y_normal, test_size=0.2)
    # clf = OneClassSVM(gamma=0.001, nu=0.1).fit(X_train)
    # y_pred_train = clf.predict(X_test)
    #
    # print("Training check")
    #
    # print(np.sum(y_pred_train == 1)/len(y_pred_train))
    #
    # X_train, X_test, y_train, y_test = train_test_split(DatasetOutliers, y_abnormal, test_size=0.2)
    # y_pred_test = clf.predict(DatasetOutliers)
    #
    # print("TEst check")
    #
    # print(np.sum(y_pred_test == -1)/ len(DatasetOutliers))


if __name__ == "__main__":
    labels = shapeElab()
    # intensityElab()
    # totalProcessing()