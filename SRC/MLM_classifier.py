import math
from math import *

import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

from Process_Data import processData

# from src.Classification.process_data import processData

input_y_test_file = "../DATA/input/data_test.xlsx"


class MLM_classifier():
    def __init__(self, Classification, u_recommender, train_set_train=None, train_set_test=None):
        self.Classification = Classification
        self.u_recommender = u_recommender
        # self.input_cl_file = input_cl_file
        self.train_set_train = train_set_train
        self.train_set_test = train_set_test

    def MLM_classifier(self):

        # Dung sklearn.kNN new 30/05/2021
        X_train = self.train_set_train['matrix_tf_idf']
        y_train = np.zeros((1350, 5))
        X_test = self.train_set_test['matrix_tf_idf']
        train_set = processData(input_dl_file=input_y_test_file).readFilleDl()
        y_test = np.zeros((150, 5))

        for u in self.Classification:
            for labels in self.Classification[u]:
                y_train[u - 1][labels] = 1

        for u in train_set['items_seen_by_user']:
            for labels in train_set['items_seen_by_user'][u]:
                y_test[u - 1][labels] = 1
        # Create KNN Classifier
        knn = KNeighborsClassifier(n_neighbors=1)

        # Train the model using the training sets
        knn.fit(X_train, y_train)

        l = knn.kneighbors(X_test, n_neighbors=1, return_distance=False)
        # X[l].ravel()

        # Predict the response for test dataset
        predictions = knn.predict(X_test)

        return predictions

    # Do do euclid
    def euclid_distance(self, x, y):
        return sqrt(sum(pow(a - b, 2) for a, b in zip(x, y)))

    # Do do cosin
    def cosineSimilarity(self, X, Y):
        numeator = sum(a * b for a, b in zip(X, Y))
        denominator = self.square_rooted(X) * self.square_rooted(Y)
        return numeator / float(denominator)

    def square_rooted(self, x):
        return round(math.sqrt(sum([a * a for a in x])), 5)

    # Do do manhattan
    def manhattan_distance(self, x, y):
        return sum(abs(a - b) for a, b in zip(x, y))

    # Ham danh gia mo hinh
    def evalue(input_file_test, classi):

        train_set = processData(input_dl_file=input_file_test).readFilleDl()
        y_test = np.zeros((len(train_set['users']), 5))
        for u in train_set['items_seen_by_user']:
            for labels in train_set['items_seen_by_user'][u]:
                y_test[u - 1][labels] = 1

        predictions = classi

        # Danh gia mo hinh
        recal = metrics.classification_report(y_test, predictions)
        hamming_loss = metrics.hamming_loss(y_test, predictions)
        zero_one_loss = metrics.zero_one_loss(y_test, predictions)
        coverage_error = metrics.coverage_error(y_test, predictions)
        label_ranking_loss = metrics.label_ranking_loss(y_test, predictions)
        average_precision_score = metrics.average_precision_score(y_test, predictions)
        print(recal)
        print('----------------------------------------')
        print('hamming_los:', hamming_loss)
        print('zero_one_loss:', zero_one_loss)
        print('coverage_error:', coverage_error)
        print('label_ranking_loss:', label_ranking_loss)
        print('average_precision_score:', average_precision_score)
        print("Accuracy:", metrics.accuracy_score(y_test, predictions))

        return "Done!"
