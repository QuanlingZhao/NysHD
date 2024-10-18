
from sklearn.model_selection import train_test_split
import re

def get_data(test_frac, random_state):
    file = "./data/promoters.data"
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
    train_x, test_x, train_y, test_y = train_test_split(sequences, labels, test_size=test_frac, random_state=random_state)

    return  train_x, test_x, train_y, test_y













