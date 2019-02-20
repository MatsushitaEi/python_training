from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.datasets import fetch_mldata
from sklearn.neighbors import KNeighborsClassifier
from numba import jit

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

mnist = fetch_mldata('MNIST original')
mnist_X, mnist_y = shuffle(mnist.data, mnist.target, random_state=42)

mnist_X = mnist_X / 255.0

train_X, test_X, train_y, test_y = train_test_split(mnist_X, mnist_y,
                                                    test_size=0.2,
                                                    random_state=42)
train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y,
                                                      test_size=0.2,
                                                      random_state=42)


@jit
def calc_knn(clf, x, y):
    clf.fit(x, y)
    return clf


# クラスタリングの数
n_neighbors = 10
clf = KNeighborsClassifier(n_neighbors=n_neighbors)
clf = calc_knn(clf, train_X, train_y)

print(clf.predict(test_X[:100]))
print(test_y[:100])
