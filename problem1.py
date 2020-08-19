import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

class Perceptron(object):

    def __init__(self, learning_rate=0.01, n_iters=50):
        self.lr = learning_rate
        self.n_iter = n_iters

    def train(self, X, y, out):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        outputFile = open(str(out), 'w')
        for _ in range(self.n_iter):
            errors = 0
            for xi, label in zip(X, y):
                update = self.lr * (label - self.predict(xi))
                self.w_[0:2] += update * xi
                self.w_[2] += update
                # print(self.w_)
                # print(self.w_[0:2])
                # print(self.w_[2])
                errors += int(update != 0.0)

            self.errors_.append(errors)
            outputFile.write(str(round(self.w_[0], 2)) + ',' + str(round(self.w_[1], 2)) + ',' + str(round(self.w_[2], 2)) + '\n')
            if errors == 0:
                outputFile.close()
                return self
        outputFile.close()
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[0:2]) + self.w_[2]

    def predict(self, X):
        return np.where(self.net_input(X) > 0, 1, -1)

def main():
    input1 = pd.read_csv('C:/Users/vlad1/Desktop/AI/Week 7/Assignment 3/input1.csv', header=None)

    # outputFile = open('%s', out, 'w')
    y = input1.iloc[:, 2]
    X = input1.iloc[:, [0, 1]].values
    ppn = Perceptron(0.01, 50)
    ppn.train(X, y, 'output1.csv')

    plot_decision_regions(np.array(X), np.array(y), clf=ppn)
    plt.title('Perceptron')
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.show()


if __name__ == '__main__':
    main()