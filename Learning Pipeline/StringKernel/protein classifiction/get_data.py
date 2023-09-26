
from sklearn.model_selection import train_test_split

def get_data(test_frac, random_state):
    class0_file = "./data/bioinfo.imtech.res.in_servers_rhodopred_dataset_Actinorhodopsin.txt"
    class1_file = "./data/bioinfo.imtech.res.in_servers_rhodopred_dataset_Bacteriorhodopsin.txt"
    class2_file = "./data/bioinfo.imtech.res.in_servers_rhodopred_dataset_Halorhodopsin.txt"
    class3_file = "./data/bioinfo.imtech.res.in_servers_rhodopred_dataset_Proteorhodopsin.txt"
    class4_file = "./data/bioinfo.imtech.res.in_servers_rhodopred_dataset_Sensory-rhodopsin.txt"
    class5_file = "./data/bioinfo.imtech.res.in_servers_rhodopred_dataset_Xanthorhodopsin.txt"

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

    X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=test_frac, random_state=random_state)

    return  X_train, X_test, y_train, y_test















