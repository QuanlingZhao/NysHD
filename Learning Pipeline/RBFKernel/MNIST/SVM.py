from mnist import MNIST
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import math
import random
import copy
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC



if __name__ == "__main__":
    data = MNIST('data')
    train_x, train_y = data.load_training()
    test_x, test_y = data.load_testing()

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    clf = SVC(C=100,gamma=0.01,kernel='rbf')
    clf.fit(train_x, train_y)

    y_pred = clf.predict(test_x)

    print('accuracy score: ' + str(accuracy_score(test_y, y_pred)))








