
import pickle
import numpy as np
import torch
from gen_kernel_matrix import *
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import copy


def encode(dimension,dataset,precomputed_kernel):
    if precomputed_kernel== True:
        with open('KernelMatrices/'+dataset+'.pkl', 'rb') as input:
            all_data = pickle.load(input)
    else:
        all_data = gen_kernel_matrix(dataset,False)
    dataset = all_data['dataset']
    num_landmark = all_data['num_landmark']
    landmark_kernel_matrix = all_data['landmark_kernel_matrix']
    train_kernel_matrix = all_data['train_kernel_matrix']
    test_kernel_matrix = all_data['test_kernel_matrix']
    train_label = all_data['train_label']
    test_label = all_data['test_label']
    train_num = all_data['train_num']
    test_num = all_data['test_num']
    eigen_values, eigen_vectors = np.linalg.eigh(landmark_kernel_matrix)
    eigen_values = eigen_values.real
    eigen_values[eigen_values<=0] = 1e-15
    eigen_vectors = eigen_vectors.real
    diag = np.diag(1 / np.sqrt(eigen_values), k=0)
    eigen = eigen_vectors.T
    proj = np.random.uniform(low=-1, high=1, size=(dimension,num_landmark))
    proj_norm = proj / np.linalg.norm(proj,axis=1)[:,None]
    encoding_matrix = np.matmul(proj_norm,np.matmul(diag,eigen))
    if dataset == "MNIST" or dataset == "FashionMNIST":
        train_encodings = (np.sqrt(np.pi / (2))) * (np.sqrt(1 / (dimension))) * np.sign(np.float32(train_kernel_matrix) @ np.float32(encoding_matrix).T)
        test_encodings = (np.sqrt(np.pi / (2))) * (np.sqrt(1 / (dimension))) * np.sign(np.float32(test_kernel_matrix) @ np.float32(encoding_matrix).T)
    else:
        train_encodings = (np.sqrt(np.pi / (2))) * (np.sqrt(1 / (dimension))) * np.sign(train_kernel_matrix @ encoding_matrix.T)
        test_encodings = (np.sqrt(np.pi / (2))) * (np.sqrt(1 / (dimension))) * np.sign(test_kernel_matrix @ encoding_matrix.T)

    return train_encodings, test_encodings, train_label, test_label, train_num, test_num



class hd_model:
    def __init__(self,dataset,dimension,lr,epoch,precomputed_kernel):
        assert dataset in ['MNIST','FashionMNIST','ENZYMES','NCI1','Protein','SMS','MUTAG','NCI109','DD','BZR','COX2','Mutagenicity','splice','promoter']
        self.dataset = dataset
        self.dimension = dimension
        self.lr = lr
        self.epoch = epoch
        self.precomputed_kernel = precomputed_kernel
        self.train_encodings,self.test_encodings,self.train_label,self.test_label,self.train_num,self.test_num=encode(self.dimension,dataset,self.precomputed_kernel)
        if self.dataset == "MNIST":
            self.class_hvs = np.zeros((10,self.dimension))
            self.class_hvs = np.float32(self.class_hvs)
        if self.dataset == "FashionMNIST":
            self.class_hvs = np.zeros((10,self.dimension))
            self.class_hvs = np.float32(self.class_hvs)
        if self.dataset == "ENZYMES":
            self.class_hvs = np.zeros((6,self.dimension))
        if self.dataset == "NCI1":
            self.class_hvs = np.zeros((2,self.dimension))
        if self.dataset == "Protein":
            self.class_hvs = np.zeros((6,self.dimension))
        if self.dataset == "SMS":
            self.class_hvs = np.zeros((2,self.dimension))
        if self.dataset == "MUTAG":
            self.class_hvs = np.zeros((2,self.dimension))
        if self.dataset == "NCI109":
            self.class_hvs = np.zeros((2,self.dimension))
        if self.dataset == "DD":
            self.class_hvs = np.zeros((2,self.dimension))
        if self.dataset == "BZR":
            self.class_hvs = np.zeros((2,self.dimension))
        if self.dataset == "COX2":
            self.class_hvs = np.zeros((2,self.dimension))
        if self.dataset == "Mutagenicity":
            self.class_hvs = np.zeros((2,self.dimension))
        if self.dataset == "splice":
            self.class_hvs = np.zeros((3,self.dimension))
        if self.dataset == "promoter":
            self.class_hvs = np.zeros((2,self.dimension))
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






































