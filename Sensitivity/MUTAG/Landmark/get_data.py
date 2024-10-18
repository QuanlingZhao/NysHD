from grakel.datasets import fetch_dataset
import numpy as np
from sklearn.model_selection import train_test_split




def get_data():
    MUTAG = fetch_dataset("MUTAG",verbose=False,download_if_missing=True,prefer_attr_nodes=True)
    G, y = MUTAG.data, MUTAG.target
    y[y==-1] = 0
    G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=0.2, random_state=42)
    return G_train, G_test, y_train, y_test






