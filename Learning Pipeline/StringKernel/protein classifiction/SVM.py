from gappy_kernel import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from Bio.Seq import Seq
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve, precision_score
from Bio import SeqIO
from scipy.sparse import coo_matrix, vstack
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import random
from sklearn.metrics.pairwise import cosine_similarity
from get_data import *
from sklearn.svm import SVC




if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_data(0.2,42)

    k = 3
    g = 0
    t = 2

    train_features = gappypair_kernel(X_train, k=k, g=g, t=t, include_flanking=True, gapDifferent = False, sparse = True).toarray()
    train_kernel_matrix = np.matmul(train_features, np.transpose(train_features))

    test_features = gappypair_kernel(X_test, k=k, g=g, t=t, include_flanking=True, gapDifferent = False, sparse = True).toarray()
    test_kernel_matrix = np.matmul(test_features, np.transpose(train_features))

    svc = SVC(kernel='precomputed')
    svc.fit(train_kernel_matrix, y_train)

    y_pred = svc.predict(test_kernel_matrix)
    print('accuracy score: ' + str(accuracy_score(y_test, y_pred)))








