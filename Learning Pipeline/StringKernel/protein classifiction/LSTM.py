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
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



class LSTMClassifier(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length):
        super(LSTMClassifier, self).__init__()
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        self.word_embeddings = nn.Embedding(vocab_size, embedding_length, padding_idx=0)
        self.lstm = nn.LSTM(embedding_length, hidden_size,num_layers = 2)
        self.fc1 = nn.Linear(hidden_size, 50)
        self.fc2 = nn.Linear(50,output_size)

        self.activation = nn.ReLU()

    
    def forward(self, input_sentence):
        input = self.word_embeddings(input_sentence)
        input = input.permute(1, 0, 2)
        output, _ = self.lstm(input)
        final_output = self.fc2(self.activation((self.fc1(output[-1,:,:]))))
        return final_output



class LSTM_model:
    def __init__(self, train_x, train_y, test_x, test_y,lr):
        self.lr = lr

        self.model = LSTMClassifier(64,6,100,27,100)
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

        self.max_length = 0
        for i in range(len(self.train_x)):
             if len(self.train_x[i]) > self.max_length:
                  self.max_length = len(self.train_x[i])


    def translate(self,string):
        mapping = {"A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7,"H":8,"I":9,"J":10,"K":11,"L":12,"M":13,"N":14,
                    "O":15,"P":16,"Q":17,"R":18,"S":19,"T":20,"U":21,"V":22,"W":23,"X":24,"Y":25,"Z":26}
        l = []
        for i in range(len(string)):
            l.append(mapping[string[i]])
        l = l + ((self.max_length - len(l)) * [0])
        return l

    def train(self):
        assert len(self.train_x) == len(self.train_y)
        num_batch = math.floor(len(self.train_x) / 64)

        #for name, param in self.model.named_parameters():
        #    if param.requires_grad:
        #        print(name, param.data)
        #    else:
        ##        print('==========================')
         #       print("No Grad: " + str(name))
         #       print('==========================')

        optim = torch.optim.SGD(self.model.parameters(),lr=self.lr)
        for epoch in range(100):
            print(epoch)
            epoch_loss = 0
            for i in range(num_batch):
                x = self.train_x[i*64:(i+1)*64]
                label = self.train_y[i*64:(i+1)*64]
                for idx in range(len(x)):
                    x[idx] = self.translate(x[idx])
                x = torch.tensor(x)
                label = torch.tensor(label)

                out = self.model.forward(x)

                loss = F.cross_entropy(out,label)
                loss.backward()
                optim.step()

                epoch_loss += loss.item()
            print(epoch_loss)


    
    def test(self):
        assert len(self.train_x) == len(self.train_y)

        num_batch = math.floor(len(self.test_x) / 64)

        preds = []
        labels = []

        for i in range(num_batch):
            x = self.train_x[i*64:(i+1)*64]
            label = self.train_y[i*64:(i+1)*64]
            for idx in range(len(x)):
                x[idx] = self.translate(x[idx])
            x = torch.tensor(x)
            label = torch.tensor(label)

            out = self.model.forward(x)

            preds += torch.argmax(out, dim=1).tolist()
            labels += label.tolist()

        print(preds)
        print(labels)
        
        print("================================")
        print(accuracy_score(labels, preds))
        print(f1_score(labels, preds, average="weighted"))
        print("================================")







if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_data(0.1,42)

    model = LSTM_model(X_train, y_train, X_test, y_test,0.0001)

    model.train()
    model.test()


'''
if __name__ == "__main__":
    train_x, _, _, _ = get_data(0.1,42)

    encoder = hd_adaptive_encoder(train_x,200,3,0,2,10000)

    encs = []
    for i in range(25):
        encs.append(encoder.adaptive_encode(train_x[i]))
    
    encs = np.array(encs)

    a = (encs @ encs.T)

    _b = gappypair_kernel(train_x[0:25], k=3, g=0, t=2, include_flanking=True, gapDifferent = False, sparse = True).toarray()
    b = _b @ _b.T

    _k = gappypair_kernel(encoder.landmark, k=3, g=0, t=2, include_flanking=True, gapDifferent = False, sparse = True).toarray()
    k = _k @ _k.T
    eigen_values, eigen_vectors = np.linalg.eigh(k)
    eigen_values = eigen_values.real
    eigen_values[eigen_values<=0] = 1e-15
    eigen_vectors = eigen_vectors.real
    diag = np.diag(1 / np.sqrt(eigen_values), k=0)    
    eigen = eigen_vectors.T
    nystrom = _b @ _k.T
    c = nystrom @ eigen.T @ diag @ diag @ eigen @ nystrom.T

    plt.imshow(a)
    plt.colorbar()
    plt.show()

    plt.imshow(b)
    plt.colorbar()
    plt.show()

    plt.imshow(c)
    plt.colorbar()
    plt.show()
'''
























