import torch
import numpy as np
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import re
from mnist import MNIST
from sklearn.preprocessing import Normalizer

#########################################################
#########################################################
#########################################################
def sparse_stochastic_graph(G):
    """
    Returns a sparse adjacency matrix of the graph G.
    The values indicate the probability of leaving a vertex.
    This means that each column sums up to one.
    """
    rows, columns = G.edge_index
    # Calculate the probability for each column
    values_per_column = 1.0 / torch.bincount(columns, minlength=G.num_nodes)
    values_per_node = values_per_column[columns]
    size = (G.num_nodes, G.num_nodes)
    return torch.sparse_coo_tensor(G.edge_index, values_per_node, size)
def pagerank(G, alpha=0.85, max_iter=100):
    N = G.num_nodes
    M = sparse_stochastic_graph(G) * alpha
    v = torch.full((N,), 1 / N)
    p = torch.full((N,), (1 - alpha) / N)
    for _ in range(max_iter):
        v = M @ v + p
    return v
def to_undirected(edge_index):
    """
    Returns the undirected edge_index
    [[0, 1], [1, 0]] will result in [[0], [1]]
    """
    edge_index = edge_index.sort(dim=0)[0]
    edge_index = torch.unique(edge_index, dim=1)
    return edge_index
def inverse_permutation(perm):
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.size(0), device=perm.device)
    return inv
def random_bipolar_hv(dim):
    return np.random.choice([-1, 1], size=(dim), p=[0.5, 0.5])
class Encoder(nn.Module):
    def __init__(self, size):
        super(Encoder, self).__init__()
        self.node_ids = [random_bipolar_hv(10000) for i in range(size)]
    def forward(self, x):
        pr = pagerank(x, max_iter=10)
        pr_argsort = inverse_permutation(torch.argsort(pr))
        node_id_hvs = [self.node_ids[j] for j in pr_argsort]
        row, col = to_undirected(x.edge_index)
        hvs = [node_id_hvs[s] * node_id_hvs[t] for s, t in zip(row, col)]
        return sum(hvs)
    
class hd_n_gram_encoder:
    def __init__(self,dim,n):
        self.dim = dim
        self.n = n
        self.init_codebook()

    def init_codebook(self):
        alphabet = {}
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
            alphabet[letter] = np.random.randint(2, size=self.dim)
            alphabet[letter][alphabet[letter]<=0] = -1
        self.alphabet = alphabet
    
    def n_gram_encode(self,sequence):
        sequence_length = len(sequence)
        enc = np.zeros(self.dim)
        for i in range(sequence_length - self.n + 1):
            n_gram = sequence[i:i+self.n]
            gram_enc = self.alphabet[n_gram[0]]
            for letter in range(self.n-1):
                np.multiply(gram_enc, np.roll(self.alphabet[n_gram[letter+1]],letter))
            enc += (gram_enc).astype('i')
        return np.sign((enc).astype('i'))
#########################################################
#########################################################
#########################################################

def encode_ENZYMES():
    dataset = "ENZYMES"
    graphs = TUDataset("./datasets/ENZYMES", dataset)
    max_graph_size = max(g.num_nodes for g in graphs)
    encode = Encoder(max_graph_size)
    all_enc = []
    all_y = []
    for g in graphs:
        samples_hv = encode(g)
        all_enc.append(samples_hv)
        all_y.append(g.y.item())
    G_train, G_test, y_train, y_test = train_test_split(all_enc, all_y, test_size=0.2, random_state=42)
    return np.array(G_train), np.array(G_test), np.array(y_train), np.array(y_test), len(G_train), len(G_test)

def encode_NCI1():
    dataset = "NCI1"
    graphs = TUDataset("./datasets/NCI1", dataset)
    max_graph_size = max(g.num_nodes for g in graphs)
    encode = Encoder(max_graph_size)
    all_enc = []
    all_y = []
    for g in graphs:
        samples_hv = encode(g)
        all_enc.append(samples_hv)
        all_y.append(g.y.item())
    G_train, G_test, y_train, y_test = train_test_split(all_enc, all_y, test_size=0.1, random_state=42)
    return np.array(G_train), np.array(G_test), np.array(y_train), np.array(y_test), len(G_train), len(G_test)


def encode_BZR():
    dataset = "BZR"
    graphs = TUDataset("./datasets/BZR", dataset)
    max_graph_size = max(g.num_nodes for g in graphs)
    encode = Encoder(max_graph_size)
    all_enc = []
    all_y = []
    for g in graphs:
        samples_hv = encode(g)
        all_enc.append(samples_hv)
        all_y.append(g.y.item())
    G_train, G_test, y_train, y_test = train_test_split(all_enc, all_y, test_size=0.1, random_state=42)
    return np.array(G_train), np.array(G_test), np.array(y_train), np.array(y_test), len(G_train), len(G_test)


def encode_DD():
    dataset = "DD"
    graphs = TUDataset("./datasets/DD", dataset)
    max_graph_size = max(g.num_nodes for g in graphs)
    encode = Encoder(max_graph_size)
    all_enc = []
    all_y = []
    for g in graphs:
        samples_hv = encode(g)
        all_enc.append(samples_hv)
        all_y.append(g.y.item())
    G_train, G_test, y_train, y_test = train_test_split(all_enc, all_y, test_size=0.1, random_state=42)
    return np.array(G_train), np.array(G_test), np.array(y_train), np.array(y_test), len(G_train), len(G_test)


def encode_COX2():
    dataset = "COX2"
    graphs = TUDataset("./datasets/COX2", dataset)
    max_graph_size = max(g.num_nodes for g in graphs)
    encode = Encoder(max_graph_size)
    all_enc = []
    all_y = []
    for g in graphs:
        samples_hv = encode(g)
        all_enc.append(samples_hv)
        all_y.append(g.y.item())
    G_train, G_test, y_train, y_test = train_test_split(all_enc, all_y, test_size=0.1, random_state=42)
    return np.array(G_train), np.array(G_test), np.array(y_train), np.array(y_test), len(G_train), len(G_test)


def encode_MUTAG():
    dataset = "MUTAG"
    graphs = TUDataset("./datasets/MUTAG", dataset)
    max_graph_size = max(g.num_nodes for g in graphs)
    encode = Encoder(max_graph_size)
    all_enc = []
    all_y = []
    for g in graphs:
        samples_hv = encode(g)
        all_enc.append(samples_hv)
        all_y.append(g.y.item())
    G_train, G_test, y_train, y_test = train_test_split(all_enc, all_y, test_size=0.1, random_state=42)
    return np.array(G_train), np.array(G_test), np.array(y_train), np.array(y_test), len(G_train), len(G_test)



def encode_Mutagenicity():
    dataset = "Mutagenicity"
    graphs = TUDataset("./datasets/Mutagenicity", dataset)
    max_graph_size = max(g.num_nodes for g in graphs)
    encode = Encoder(max_graph_size)
    all_enc = []
    all_y = []
    for g in graphs:
        samples_hv = encode(g)
        all_enc.append(samples_hv)
        all_y.append(g.y.item())
    G_train, G_test, y_train, y_test = train_test_split(all_enc, all_y, test_size=0.1, random_state=42)
    return np.array(G_train), np.array(G_test), np.array(y_train), np.array(y_test), len(G_train), len(G_test)



def encode_NCI109():
    dataset = "NCI109"
    graphs = TUDataset("./datasets/NCI109", dataset)
    max_graph_size = max(g.num_nodes for g in graphs)
    encode = Encoder(max_graph_size)
    all_enc = []
    all_y = []
    for g in graphs:
        samples_hv = encode(g)
        all_enc.append(samples_hv)
        all_y.append(g.y.item())
    G_train, G_test, y_train, y_test = train_test_split(all_enc, all_y, test_size=0.1, random_state=42)
    return np.array(G_train), np.array(G_test), np.array(y_train), np.array(y_test), len(G_train), len(G_test)





def encode_Protein():
    class0_file = "./datasets/Protein/bioinfo.imtech.res.in_servers_rhodopred_dataset_Actinorhodopsin.txt"
    class1_file = "./datasets/Protein/bioinfo.imtech.res.in_servers_rhodopred_dataset_Bacteriorhodopsin.txt"
    class2_file = "./datasets/Protein/bioinfo.imtech.res.in_servers_rhodopred_dataset_Halorhodopsin.txt"
    class3_file = "./datasets/Protein/bioinfo.imtech.res.in_servers_rhodopred_dataset_Proteorhodopsin.txt"
    class4_file = "./datasets/Protein/bioinfo.imtech.res.in_servers_rhodopred_dataset_Sensory-rhodopsin.txt"
    class5_file = "./datasets/Protein/bioinfo.imtech.res.in_servers_rhodopred_dataset_Xanthorhodopsin.txt"
    files = [class0_file,class1_file,class2_file,class3_file,class4_file,class5_file]
    sequences = []
    labels = []
    for num in range(len(files)):
        file = files[num]
        io = open(file, 'r')
        Lines = io.readlines()
        for line in Lines:
            seq = line.strip().split("::")
            if len(seq) != 2:
                continue
            seq = seq[1]
            sequences.append(seq)
            labels.append(num)
    encoder = hd_n_gram_encoder(10000, 3)
    all_enc = []
    for seq in sequences:
        all_enc.append(encoder.n_gram_encode(seq))
    X_train, X_test, y_train, y_test = train_test_split(all_enc, labels, test_size=0.2, random_state=42)
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test), len(X_train), len(X_test)


def encode_SMS():
    files = ["./datasets/SMS/SMSSpamCollection"]
    sequences = []
    labels = []
    for num in range(len(files)):
        file = files[num]
        io = open(file, 'r')
        Lines = io.readlines()
        for line in Lines:
            seq = line.strip().split("\t")
            if len(seq) != 2:
                continue
            seq[1] = re.sub('[^A-Za-z0-9]+', '', seq[1]).upper()
            sequences.append(seq[1])
            labels.append(1 if seq[0] == 'ham' else 0)
    encoder = hd_n_gram_encoder(10000, 3)
    all_enc = []
    for seq in sequences:
        all_enc.append(encoder.n_gram_encode(seq))
    X_train, X_test, y_train, y_test = train_test_split(all_enc, labels, test_size=0.2, random_state=42)
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test), len(X_train), len(X_test)



def encode_splice():
    file = "./datasets/splice/splice.data"
    sequences = []
    labels = []
    io = open(file, 'r')
    Lines = io.readlines()
    for line in Lines:
        parse = line.strip().split(",")
        parse[2] = re.sub('[^A-Za-z0-9]+', '', parse[2]).upper()
        sequences.append(parse[2])
        if parse[0] == 'EI':
            labels.append(0)
        elif parse[0] == 'IE':
            labels.append(1)
        elif parse[0] == 'N':
            labels.append(2)
        else:
            print('Error')
            print(parse[0])
            exit()
    encoder = hd_n_gram_encoder(10000, 3)
    all_enc = []
    for seq in sequences:
        all_enc.append(encoder.n_gram_encode(seq))
    X_train, X_test, y_train, y_test = train_test_split(all_enc, labels, test_size=0.2, random_state=42)
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test), len(X_train), len(X_test)



def encode_promoter():
    file = "./datasets/promoter/promoters.data"
    sequences = []
    labels = []
    io = open(file, 'r')
    Lines = io.readlines()
    for line in Lines:
        parse = line.strip().split(",")
        parse[2] = re.sub('[^A-Za-z0-9]+', '', parse[2]).upper()
        sequences.append(parse[2])
        if parse[0] == '+':
            labels.append(0)
        elif parse[0] == '-':
            labels.append(1)
        else:
            print('Error')
            print(parse[0])
            exit()
    encoder = hd_n_gram_encoder(10000, 3)
    all_enc = []
    for seq in sequences:
        all_enc.append(encoder.n_gram_encode(seq))
    X_train, X_test, y_train, y_test = train_test_split(all_enc, labels, test_size=0.2, random_state=42)
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test), len(X_train), len(X_test)




def encode_MNIST(linear):
    data = MNIST('datasets/MNIST')
    train_x, train_y = data.load_training()
    test_x, test_y = data.load_testing()
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    if linear:
        projection = np.random.normal(0, 1, size=(28*28, 10000))
        X_train = np.sign(np.float32(train_x) @ np.float32(projection))
        X_test = np.sign(np.float32(test_x) @ np.float32(projection))
        return X_train, X_test, train_y, test_y, len(X_train), len(X_test)
    if not linear:
        scaler = Normalizer(norm='l2').fit(train_x)
        train_x = scaler.transform(train_x)
        test_x = scaler.transform(test_x)
        bases = torch.empty(28*28, 10000)
        bases = bases.normal_(0, 1)
        bases = bases.numpy()
        bias = np.random.uniform(low=0, high=2*np.pi, size=(10000))
        X_train = np.cos((np.float32(train_x) @ np.float32(bases))+np.float32(bias))
        X_test = np.cos((np.float32(test_x) @ np.float32(bases))+np.float32(bias))
        return X_train, X_test, train_y, test_y, len(X_train), len(X_test)
    

def encode_FashionMNIST(linear):
    data = MNIST('datasets/FashionMNIST')
    train_x, train_y = data.load_training()
    test_x, test_y = data.load_testing()
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    if linear:
        projection = np.random.normal(0, 1, size=(28*28, 10000))
        X_train = np.sign(np.float32(train_x) @ np.float32(projection))
        X_test = np.sign(np.float32(test_x) @ np.float32(projection))
        return X_train, X_test, train_y, test_y, len(X_train), len(X_test)
    if not linear:
        scaler = Normalizer(norm='l2').fit(train_x)
        train_x = scaler.transform(train_x)
        test_x = scaler.transform(test_x)
        bases = torch.empty(28*28, 10000)
        bases = bases.normal_(0, 1)
        bases = bases.numpy()
        bias = np.random.uniform(low=0, high=2*np.pi, size=(10000))
        X_train = np.cos((np.float32(train_x) @ np.float32(bases))+np.float32(bias))
        X_test = np.cos((np.float32(test_x) @ np.float32(bases))+np.float32(bias))
        return X_train, X_test, train_y, test_y, len(X_train), len(X_test)








if __name__ == '__main__':
    #encode_SMS()
    #encode_Protein()
    #encode_MNIST(True)
    #encode_MNIST(False)
    #encode_FashionMNIST(True)
    #encode_FashionMNIST(False)
    #encode_ENZYMES()
    #encode_NCI1()
    #encode_COX2()
    #encode_DD()
    #encode_BZR()
    #encode_NCI109()
    #encode_MUTAG()
    #encode_Mutagenicity()
    #encode_splice()
    #encode_promoter()
    pass