# Requirements
import numpy as np
import math


def basic_sigmoid(x):
    s = 1 / (1 + math.exp(-x))
    return s
    ## End of Function ##


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s
    ## End of Function ##


def sigmoid_derivative(x):
    s = sigmoid(x)
    ds = s * (1 - s) 
    return ds
    ## End of Function ##


def image2vector(image):
    v = image.reshape(image.shape[0] * image.shape[1] * image.shape[2], 1)
    return v
    ## End of Function ##


def normalize_rows(x):
    x_norm = np.linalg.norm(x, ord = 2, axis = 1, keepdims = True)
    x = x / x_norm
    return x
    ## End of Function ##


def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis = 1, keepdims = True)
    s = x_exp / x_sum
    return s
    ## End of Function ##


def L1(yhat, y):
    loss = np.sum(np.absolute(y - yhat))
    return loss
    ## End of Function ##


def L2(yhat, y):
    loss = np.sum(np.power(y - yhat, 2))
    return loss
    ## End of Function ##