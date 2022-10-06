

from collections import Counter
import csv
import numpy as np

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_score
import time
from sklearn.feature_selection import RFECV
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
import sklearn
def takeDatasetFromCSV(path):

    Dataset = []
    with open(str(path)) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            Dataset.append(np.asfarray(row))

    ##Delete the last row 'cause it may be corrupted
    Dataset.pop(len(Dataset) - 1)
    Dataset = np.vstack(Dataset)
    # print(np.asmatrix(Dataset))

    #delete the first column cause it's the ID
    Dataset = np.delete(Dataset, 0, 1)

    return Dataset


def ExtractLabels(path):

    labels = []
    with open(str(path)) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            labels = row

    # print(labels)
    labels.pop(len(labels) - 1)
    labels = np.asfarray(labels)
    return labels


def classifier(Dataset, y):


    names = ["Decision Tree", "Random Forest", "AdaBoost"]

    classifiers = [
        DecisionTreeClassifier(max_depth=10),
        RandomForestClassifier(max_depth=10, n_estimators=10, max_features=1),
        AdaBoostClassifier(n_estimators=50)]

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


        Dataset = rfecv.transform(Dataset)
        scores = cross_val_score(clf, Dataset, y, cv=10)
        # # print("training time:", round(time.time() - t0, 3), "s"  )# the time would be round to 3 decimal in seconds
        print(names[index])
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 1.96/np.sqrt(10)))
        index += 1

def ExtractPCAFeatures(Dataset, number_of_components = 10):

    pca = PCA(n_components=number_of_components, svd_solver='auto')
    pca.fit(np.nan_to_num(Dataset))
    print("The best component")
    print( pca.explained_variance_ratio_)
    # print(pca.singular_values_)
    DatasetShape = pca.transform(np.nan_to_num(Dataset))

    return DatasetShape, pca

#
# def rebalanceDataset(Dataset, ):
#


def shapeClassification():
    DatasetShape = takeDatasetFromCSV("featureExtraction1/accelleration_pushed_cones_inverted/shapeFeatures.csv")
    # DatasetIntensity = takeDatasetFromCSV("featureExtraction1/accelleration_pushed/shapeFeatures.csv")
    Labels = ExtractLabels("featureExtraction1/accelleration_pushed_cones_inverted/labelsCones.csv")

    print('Original dataset shape %s' % Counter(Labels))

    DatasetShape = preprocessing.scale(DatasetShape, axis=0)
    # DatasetShape = sklearn.preprocessing.normalize(DatasetShape, axis=0)

    DatasetShape, pca = ExtractPCAFeatures(DatasetShape, number_of_components=25)
    i = 0
    for elem in Labels:
        if elem == 2:
            Labels[i] = 1

        i += 1

    print('Original dataset shape %s' % Counter(Labels))
    sm = SMOTE(sampling_strategy='minority', random_state=7)

    DatasetShapeResample, y = sm.fit_resample(DatasetShape, Labels)

    print('Original dataset shape %s' % Counter(y))
    classifier(DatasetShapeResample, y)

def intensityClassification():



    ##################################################################ààààà
    # DatasetShape = takeDatasetFromCSV("featureExtraction1/accelleration_pushed/shapeFeatures.csv")
    DatasetIntensity = takeDatasetFromCSV("featureExtraction1/accelleration_pushed_cones_inverted/intensityFeatures.csv")
    Labels = ExtractLabels("featureExtraction1/accelleration_pushed_cones_inverted/labelsCones.csv")

    print(len(Labels))
    print(len(DatasetIntensity))
    print('Original dataset shape %s' % Counter(Labels))
    Labels = Labels[:len(Labels) - 10]
    # del Labels[-10:]
    DatasetIntensity = preprocessing.scale(DatasetIntensity, axis=0)
    # DatasetIntensity = sklearn.preprocessing.normalize(DatasetIntensity, axis=0)
    #
    # DatasetIntensity, pca = ExtractPCAFeatures(DatasetIntensity, number_of_components=25)
    # i = 0
    # for elem in Labels:
    #     if elem == 2:
    #         Labels[i] = 1
    #
    #     i += 1
    #

    indices = [i for i, x in enumerate(Labels) if x == 0]
    Labels = Labels.tolist()
    DatasetIntensity = DatasetIntensity.tolist()
    for index in sorted(indices, reverse=True):
        del Labels[index]
        del DatasetIntensity[index]



    print('Original dataset shape %s' % Counter(Labels))
    sm = SMOTE(sampling_strategy='all', random_state=7)
    #
    DatasetShapeResample, y = sm.fit_resample(DatasetIntensity, Labels)

    print('Original dataset shape %s' % Counter(y))
    classifier(DatasetShapeResample, y)




if __name__ == "__main__":
    shapeClassification()
    intensityClassification()
