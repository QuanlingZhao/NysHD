from grakel.datasets import fetch_dataset
import numpy as np
from sklearn.model_selection import train_test_split




def get_data():
    DD = fetch_dataset("DD",verbose=False,download_if_missing=True,prefer_attr_nodes=True)
    G, y = DD.data, DD.target
    y[y==2] = 0
    G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=0.2, random_state=42)
    return G_train, G_test, y_train, y_test






