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
import copy
import math
import csv


class hd_adaptive_encoder:
    def __init__ (self,training_dataset,num_lanmark,k,g,t,dim,landmarks):
        self.training_dataset = training_dataset
        self.num_landmark = num_lanmark
        self.k = k
        self.g = g
        self.t = t
        self.dim = dim
        self.landmarks = landmarks
        self.generate_adaptive_random_projection()

    def uniform_landmark_selection(self):
        if self.landmarks != None:
            print("Preselected")
            return self.landmarks
        return random.sample(self.training_dataset,self.num_landmark)
    
    def compute_kernel_matrix(self):
        landmark_featuren = gappypair_kernel(self.landmark, k=self.k, g=self.g, t=self.t, include_flanking=True, gapDifferent = False, sparse = True).toarray()
        kernel_matrix = np.matmul(landmark_featuren, np.transpose(landmark_featuren))
        return kernel_matrix, landmark_featuren
    
    def get_unit_random_projection(self):
        #proj = np.random.uniform(low=-1, high=1, size=(self.num_landmark,self.dim))
        #proj_norm = proj / np.linalg.norm(proj,axis=1)[:,None]
        #return proj_norm.T
        #a = np.random.rand(self.dim, self.num_landmark)
        #q, r = np.linalg.qr(a)
        #return q
        proj = np.random.uniform(low=-1, high=1, size=(self.dim,self.num_landmark))
        proj_norm = proj / np.linalg.norm(proj,axis=1)[:,None]
        return proj_norm
        #return np.sign(np.random.uniform(low=-1, high=1, size=(self.dim,self.num_landmark)))

    def generate_adaptive_random_projection(self):
        self.landmark = self.uniform_landmark_selection()
        self.kernel_matrix, self.landmark_features = self.compute_kernel_matrix()
        eigen_values, eigen_vectors = np.linalg.eigh(self.kernel_matrix)
        eigen_values = eigen_values.real
        eigen_values[eigen_values<=0] = 1e-15
        eigen_vectors = eigen_vectors.real
        diag = np.diag(1 / np.sqrt(eigen_values), k=0)
        eigen = eigen_vectors.T
        self.adaptive_proj = np.matmul(diag,eigen)
        self.random_proj = self.get_unit_random_projection()
    

    def adaptive_encode(self,seq):
        features = gappypair_kernel([seq], k=self.k, g=self.g, t=self.t, include_flanking=True, gapDifferent = False, sparse = True).toarray().T
        enc = np.matmul(self.adaptive_proj,np.matmul(self.landmark_features,features))
        #enc = enc / (np.linalg.norm(enc.flatten()) + 1e-15)
        return (np.sqrt(np.pi / (2))) * (np.sqrt(1 / (self.dim))) * np.sign(np.squeeze(np.matmul(self.random_proj,enc)))



class hd_adaptive_model:
    def __init__(self, dim, num_lanmark, k, g, t, train_x, train_y, test_x, test_y,lr,landmarks):
        self.dim = dim
        self.lr = lr
        self.landmarks = landmarks

        self.encoder = hd_adaptive_encoder(train_x,num_lanmark,k,g,t,dim,landmarks)
        self.class_hvs = np.zeros((6,self.dim))
        
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

        self.train_encs = []




    def train(self):
        assert len(self.train_x) == len(self.train_y)
        for i in range(len(self.train_x)):
            if i % 500 == 0:
                print(i)
            x = self.train_x[i]
            label = self.train_y[i]    
            enc = self.encoder.adaptive_encode(x)

            similarities = cosine_similarity(enc.reshape(1, -1), self.class_hvs)[0]
            softmax = np.exp(similarities) / sum(np.exp(similarities))
            pred = np.argmax(similarities)
            
            self.class_hvs[label] += (1 - similarities[label]) * enc
            self.class_hvs[pred] -= (1 - similarities[pred]) * enc

            self.train_encs.append(enc)



    
    def test(self):
        assert len(self.test_x) == len(self.test_y)
        preds = []
        for i in range(len(self.test_x)):
            if i % 500 == 0:
                print(i)
            x = self.test_x[i]
            label = self.test_y[i]
            enc = self.encoder.adaptive_encode(x)
            similarities = cosine_similarity(enc.reshape(1, -1), self.class_hvs)[0]
            pred = np.argmax(similarities)
            preds.append(pred)
        
        print("================================")
        print(accuracy_score(self.test_y, preds))
        print(f1_score(self.test_y, preds, average="weighted"))
        print("================================")


    def retrain(self):
        for e in range(10):
            count = 0
            print(e)
            for i in range(len(self.train_encs)):
                enc = self.train_encs[i]
                label = self.train_y[i]
                similarities = cosine_similarity(enc.reshape(1, -1), self.class_hvs)[0]
                pred = np.argmax(similarities)
                if pred != label:
                    self.class_hvs[label] += self.lr * (1 - similarities[label]) * enc
                    self.class_hvs[pred] -= self.lr * (1 - similarities[pred]) * enc
                    count += 1
            print(count)





if __name__ == "__main__":
    k = 2
    g = 1
    t = 2

    X_train, X_test, y_train, y_test = get_data(0.2,42)

    landmark_featuren = gappypair_kernel(X_train,k,g,t, include_flanking=True, gapDifferent = False, sparse = True).toarray()
    real = kernel_matrix = np.matmul(landmark_featuren, np.transpose(landmark_featuren))
    print(np.count_nonzero(real))

    real_normalized = np.zeros((len(X_train),len(X_train)))
    for i in range(len(X_train)):
        for j in range(len(X_train)):
            real_normalized[i][j] = real[i][j] / (np.sqrt(real[i][i])*np.sqrt(real[j][j]))
    plt.matshow(real_normalized[0:100,0:100])
    plt.title('Normalized Kernel Matrix')
    plt.savefig('real_normalized')
    plt.close()

    real_norm = np.linalg.norm(real_normalized, ord=2)

    landmark_index = [i for i in range(len(X_train))]
    random.shuffle(landmark_index)
    result = []
    print('==========================================')
    for percentage in [5,10,15,20,25,30,35,40]:
        trials = []
        for trial in range(10):
            num_landmark = int((percentage * len(X_train)) / 100)
            landmarks =  []
            for i in range(num_landmark):
                landmarks.append(X_train[landmark_index[i]])

            s = int((len(X_train) * percentage) /100)
            print(percentage,s)
            model = hd_adaptive_model(10000, num_landmark, k, g, t, X_train, y_train, X_test, y_test,0.2,landmarks)
            model.train()

            approx = np.array(model.train_encs) @ np.array(model.train_encs).T

            if (trial==9):
                plt.matshow(approx[0:100,0:100])
                plt.title(str(percentage)+'% Landmark')
                plt.savefig(str(percentage))
                plt.close()

            diff = abs(real_norm-np.linalg.norm(approx, ord=2))
            print("Norm diff: ",diff)
            trials.append(diff)
        result.append(trials)
        print('==========================================')
    
    for i in range(8):
        print(sum(result[i])/10)
    with open('result.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(result)





















