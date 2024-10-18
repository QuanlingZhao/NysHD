from grakel.datasets import fetch_dataset
import numpy as np
from sklearn.model_selection import train_test_split




def get_data():
    ENZYMES_attr = fetch_dataset("ENZYMES", prefer_attr_nodes=True, verbose=True)
    G, y = ENZYMES_attr.data, ENZYMES_attr.target
    G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=0.2, random_state=42)
    return G_train, G_test, y_train, y_test






