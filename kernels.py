import numpy as np
from scipy.spatial.distance import cdist

#k(x, y) = <x, y>
def linear(x, y):
    return np.dot(x,y.transpose())

#k(x, y) = exp(- gamma * ||x-y||^2)
def rbf(x, y, sigma):
    pairwise_dists = cdist(x, y , metric="euclidean")
    return np.exp(- (np.power(pairwise_dists,2)) / (2 * (np.power(sigma,2))))

#k(x, y) = (<x, y> + 1 )^d
def polynomial(x, y,d):
    return (np.dot(x,y.transpose()) + 1)**d

#k(x, y) = tanh( alpha * <x, y> + c)
def sigmoid(x,y,a,c):
    return np.tanh(a * np.dot(x,y.transpose()) + c)

