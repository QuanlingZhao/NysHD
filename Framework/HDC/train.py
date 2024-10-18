
import pickle
import numpy as np
import torch
from encode import *
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


class hd_model:
    def __init__(self,dataset,lr,epoch):
        assert dataset in ['MNIST-Linear','MNIST-Nonlinear','FashionMNIST-Linear','FashionMNIST-Nonlinear','ENZYMES','NCI1','Protein','SMS','MUTAG','NCI109','DD','BZR','COX2','Mutagenicity','splice','promoter']
        self.dataset = dataset
        self.lr = lr
        self.epoch = epoch
        if self.dataset == "MNIST-Linear":
            self.class_hvs = np.float32(np.zeros((10,10000)))
            self.train_encodings,self.test_encodings,self.train_label,self.test_label,self.train_num,self.test_num=encode_MNIST(True)
        if self.dataset == "MNIST-Nonlinear":
            self.class_hvs = np.float32(np.zeros((10,10000)))
            self.train_encodings,self.test_encodings,self.train_label,self.test_label,self.train_num,self.test_num=encode_MNIST(False)
        if self.dataset == "FashionMNIST-Linear":
            self.class_hvs = np.float32(np.zeros((10,10000)))
            self.train_encodings,self.test_encodings,self.train_label,self.test_label,self.train_num,self.test_num=encode_FashionMNIST(True)
        if self.dataset == "FashionMNIST-Nonlinear":
            self.class_hvs = np.float32(np.zeros((10,10000)))
            self.train_encodings,self.test_encodings,self.train_label,self.test_label,self.train_num,self.test_num=encode_FashionMNIST(False)
        if self.dataset == "ENZYMES":
            self.class_hvs = np.zeros((6,10000))
            self.train_encodings,self.test_encodings,self.train_label,self.test_label,self.train_num,self.test_num=encode_ENZYMES()
        if self.dataset == "NCI1":
            self.class_hvs = np.zeros((2,10000))
            self.train_encodings,self.test_encodings,self.train_label,self.test_label,self.train_num,self.test_num=encode_NCI1()
        if self.dataset == "Protein":
            self.class_hvs = np.zeros((6,10000))
            self.train_encodings,self.test_encodings,self.train_label,self.test_label,self.train_num,self.test_num=encode_Protein()
        if self.dataset == "SMS":
            self.class_hvs = np.zeros((2,10000))
            self.train_encodings,self.test_encodings,self.train_label,self.test_label,self.train_num,self.test_num=encode_SMS()
        if self.dataset == "MUTAG":
            self.class_hvs = np.zeros((2,10000))
            self.train_encodings,self.test_encodings,self.train_label,self.test_label,self.train_num,self.test_num=encode_MUTAG()
        if self.dataset == "NCI109":
            self.class_hvs = np.zeros((2,10000))
            self.train_encodings,self.test_encodings,self.train_label,self.test_label,self.train_num,self.test_num=encode_NCI109()
        if self.dataset == "DD":
            self.class_hvs = np.zeros((2,10000))
            self.train_encodings,self.test_encodings,self.train_label,self.test_label,self.train_num,self.test_num=encode_DD()
        if self.dataset == "BZR":
            self.class_hvs = np.zeros((2,10000))
            self.train_encodings,self.test_encodings,self.train_label,self.test_label,self.train_num,self.test_num=encode_BZR()
        if self.dataset == "Mutagenicity":
            self.class_hvs = np.zeros((2,10000))
            self.train_encodings,self.test_encodings,self.train_label,self.test_label,self.train_num,self.test_num=encode_Mutagenicity()
        if self.dataset == "COX2":
            self.class_hvs = np.zeros((2,10000))
            self.train_encodings,self.test_encodings,self.train_label,self.test_label,self.train_num,self.test_num=encode_COX2()
        if self.dataset == "splice":
            self.class_hvs = np.zeros((3,10000))
            self.train_encodings,self.test_encodings,self.train_label,self.test_label,self.train_num,self.test_num=encode_splice()
        if self.dataset == "promoter":
            self.class_hvs = np.zeros((2,10000))
            self.train_encodings,self.test_encodings,self.train_label,self.test_label,self.train_num,self.test_num=encode_promoter()
        print('===Model initialized '+dataset+'===')

    def train(self):
        for e in range(self.epoch):
            p = np.random.permutation(len(self.train_encodings))
            self.train_encodings = self.train_encodings[p]
            self.train_label = np.array(self.train_label)[p]
            for i in range(self.train_num):
                similarities = cosine_similarity(self.train_encodings[i].reshape(1, -1), self.class_hvs)[0]
                pred = np.argmax(similarities)
                self.class_hvs[self.train_label[i]] += self.lr * (1 - similarities[self.train_label[i]]) * self.train_encodings[i]
                self.class_hvs[pred] -= self.lr * (1 - similarities[pred]) * self.train_encodings[i]
            print("Epoch:",e+1)
            acc, f1 = self.test()
        print("===Done===")

    def test(self):
        similarities = cosine_similarity(self.test_encodings, self.class_hvs)
        preds = np.argmax(similarities,axis=1)
        acc = accuracy_score(self.test_label, preds)
        f1 = f1_score(self.test_label, preds, average="weighted")
        print(acc,f1)
        return acc,f1

if __name__ == "__main__":
    pass






































