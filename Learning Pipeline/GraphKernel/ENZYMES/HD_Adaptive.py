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


class hd_adaptive_encoder:
    def __init__ (self,training_dataset,num_lanmark,dim):
        self.training_dataset = training_dataset
        self.num_landmark = num_lanmark
        self.dim = dim

        self.kernel_matrix, self.adaptive_proj = self.generate_adaptive_random_projection()


    def uniform_landmark_selection(self):
        return np.array(random.sample(self.training_dataset,self.num_landmark))
    
    def compute_kernel_matrix(self):
        self.gk = PropagationAttr()
        kernel_matrix = self.gk.fit_transform(self.landmark)

        return kernel_matrix
    
    def get_unit_random_projection(self):
        #proj = np.random.uniform(low=-1, high=1, size=(self.num_landmark,self.dim))
        #proj_norm = proj / np.linalg.norm(proj,axis=1)[:,None]
        #return proj_norm.T
        #a = np.random.rand(self.dim, self.num_landmark)
        #q, r = np.linalg.qr(a)
        #return q
        proj = np.random.uniform(low=-1, high=1, size=(self.dim,self.num_landmark))
        proj_norm = proj / np.linalg.norm(proj,axis=1)[:,None]
        return (np.sqrt(0.5 * np.pi)) * proj_norm
        #return np.sign(np.random.uniform(low=-1, high=1, size=(self.dim,self.num_landmark)))

    def generate_adaptive_random_projection(self):

        
        self.landmark = self.uniform_landmark_selection()
        kernel_matrix = self.compute_kernel_matrix()
        eigen_values, eigen_vectors = np.linalg.eigh(kernel_matrix)
        eigen_values = eigen_values.real
        eigen_values[eigen_values<=0] = 1e-15
        eigen_vectors = eigen_vectors.real
        random_proj = self.get_unit_random_projection()

        diag = np.diag(1 / np.sqrt(eigen_values), k=0)
        
        eigen = eigen_vectors.T

        adaptive_proj = np.matmul(random_proj,np.matmul(diag,eigen))
        
        return kernel_matrix, adaptive_proj
    

    def adaptive_encode(self,seq):
        features = self.gk.transform([seq]).T
        return (1 / np.sqrt(self.dim)) * np.sign(np.matmul(self.adaptive_proj,features).squeeze())



class hd_model:
    def __init__(self,train_x,train_y,test_x,test_y,num_landmark,dim,lr):
         self.train_x = train_x
         self.train_y = train_y
         self.test_x = test_x
         self.test_y =test_y
         self.num_landmark = num_landmark
         self.dim = dim
         self.lr = lr
         self.encoder = hd_adaptive_encoder(self.train_x,self.num_landmark,self.dim)
         self.class_hvs = np.zeros((10,self.dim))

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

    G_train, G_test, y_train, y_test = get_data()

    model = hd_model(G_train, y_train, G_test, y_test,200,10000,0.2)

    model.train()
    model.test()
    model.retrain()
    model.test()