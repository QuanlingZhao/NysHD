from mnist import MNIST
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import math
import torch.nn.functional as F
import torch
from sklearn.preprocessing import Normalizer
import copy




class nonlinear_random_projection:
    def __init__(self,in_dim,out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.init_projection()

    def init_projection(self):
        self.bases = torch.empty(self.out_dim, self.in_dim)
        self.bases = self.bases.normal_(0, 1)
        self.bases = self.bases.numpy()
        self.bias = np.random.uniform(low=0, high=2*np.pi, size=(self.out_dim))

    def encode(self,x):        
        return np.cos(np.matmul(self.bases,x)+self.bias)


class hd_model:
    def __init__(self,train_x,train_y,test_x,test_y,in_dim,out_dim,lr):
         self.train_x = train_x
         self.train_y = train_y
         self.test_x = test_x
         self.test_y =test_y
         self.in_dim = in_dim
         self.out_dim = out_dim
         self.lr = lr

         self.encoder = nonlinear_random_projection(self.in_dim,self.out_dim)

         self.class_hvs = np.zeros((10,self.out_dim))

         self.train_encs = []


    
    def train(self):
        assert len(self.train_x) == len(self.train_y)

        for i in range(len(self.train_x)):
            if i % 500 == 0:
                print(i)

            x = self.train_x[i]
            label = self.train_y[i]    
            enc = self.encoder.encode(x)

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
            enc = self.encoder.encode(x)
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

    data = MNIST('data')

    train_x, train_y = data.load_training()
    test_x, test_y = data.load_testing()

    scaler = Normalizer(norm='l2').fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)
    train_y = np.array(train_y)
    test_y = np.array(test_y)



    model = hd_model(train_x,train_y,test_x,test_y,28*28,10000,0.2)

    model.train()
    model.test()
    model.retrain()
    model.test()

