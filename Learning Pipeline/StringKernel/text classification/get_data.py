
from sklearn.model_selection import train_test_split
import re

def get_data(test_frac, random_state):
    files = ["./data/SMSSpamCollection"]

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

    X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=test_frac, random_state=random_state)

    return  X_train, X_test, y_train, y_test



















