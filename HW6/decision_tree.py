import numpy as np


class Node():
    def __init__(self, dim, theta):
        self.dim = dim
        self.theta = theta      # division point
        self.sign = 0           # direction
        self.left = None
        self.right = None

def gini_index(Y):
    num, = Y.shape

    pos = np.sum(Y > 0)
    neg = np.sum(Y < 0)
    
    if num == 0 or pos == 0 or neg == 0:
        return 0
    else:
        return 1 - (pos / num) ** 2 - (neg / num) ** 2

def split(X, Y):
    num, dim = X.shape
   
    min_err = np.inf
    best_dim = 0
    theta = 0

    for d in range(dim):
        index  = np.argsort(X[:, d])
        X_sort = X[index][:, d]
        Y_sort = Y[index]

        for n in range(1, num):
            Y_left = Y_sort[:n]
            Y_right = Y_sort[n:]

            err = n * gini_index(Y_left) + (num - n) * gini_index(Y_right)
            if err < min_err:
                min_err = err
                best_d = d
                theta = (X_sort[n - 1] + X_sort[n]) / 2

    X_left = X[np.where(X[:, best_d] < theta)]
    Y_left = Y[np.where(X[:, best_d] < theta)]
    X_right = X[np.where(X[:, best_d] >= theta)]
    Y_right = Y[np.where(X[:, best_d] >= theta)]

    return (X_left, Y_left), (X_right, Y_right), best_d, theta

def build(X, Y, depth):
    if X.shape[0] == 0:
        return None

    if gini_index(Y) == 0:
        node = Node(-1, -1)
        node.sign = np.sign(Y[0])
        return node
    else:
        (X_left, Y_left), (X_right, Y_right), dim, theta = split(X, Y)
        node = Node(dim, theta)
        node.left = build(X_left, Y_left, depth+1)
        node.right = build(X_right, Y_right, depth+1)

        return node

def predict(x, root):
    if root.left == None and root.right == None:
        return root.sign

    dim = root.dim
    theta = root.theta

    if x[dim] < theta:
        return predict(x, root.left)
    else:
        return predict(x, root.right)
