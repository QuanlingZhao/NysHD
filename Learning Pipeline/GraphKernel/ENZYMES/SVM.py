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
from grakel.kernels import PropagationAttr
from get_data import *
from sklearn.svm import SVC



if __name__ == "__main__":
    G_train, G_test, y_train, y_test = get_data()

    gk = PropagationAttr(t_max=1)
    train_kernel_matrix = gk.fit_transform(G_train)

    test_kernel_matrix = gk.transform(G_test)


    svc = SVC(kernel='precomputed')
    svc.fit(train_kernel_matrix, y_train)

    y_pred = svc.predict(test_kernel_matrix)
    print('accuracy score: ' + str(accuracy_score(y_test, y_pred)))








