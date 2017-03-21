import sklearn.preprocessing
import numpy as np
import pandas as pd
# import elice_utils
from scipy import linalg, stats
from functools import reduce
from math import log, sqrt

# import matplotlib as mpl
# mpl.use("Agg")
# import matplotlib.pyplot as plt

def main():
    data = pd.read_csv('./data/wine.csv', header=0)
    data = data.values
    np.random.shuffle(data)
    X = data[:, 1:]
    labels = data[:, 0]
    profile_log_likelihood(X)

def scale_data(X):
    scaler = sklearn.preprocessing.MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    mu = np.mean(X_scaled, axis=0)
    X_scaled = X_scaled - mu
    return X_scaled

def profile_log_likelihood(raw_X):
    #implement profile likelihood function
    #input description:
    #input: Data X
    #X.shape = (N, D).
    #N is number of data, D is dimension of data
    #return optimal number of component value in integer.
    X = scale_data(raw_X)
    N, D = X.shape
    l = None
    log_profile = None
    for L in range(1, D):
        u, s, v = linalg.svd(X)
        e = list(map(lambda x: x*x, s))
        eigen_upper = e[:L]
        eigen_lower = e[L:]
        upper_mean = np.mean(eigen_upper)
        lower_mean = np.mean(eigen_lower)
        var = 0

        for e in eigen_upper:
            var += (e - upper_mean) ** 2

        for e in eigen_lower:
            var += (e - lower_mean) ** 2

        var = var / D
        std = sqrt(var)
        upper_dist = stats.norm(upper_mean, std)
        lower_dist = stats.norm(lower_mean, std)

        score = 0
        for e in eigen_upper:
            score += log(upper_dist.pdf(e))

        for e in eigen_lower:
            score += log(lower_dist.pdf(e))

        print("Dimension {0}, score: {1}".format(L, score))
        if not log_profile or log_profile < score:
            log_profile = score
            l = L

    print("Final dimension: {}".format(l))
    return l

if __name__ == "__main__":
    main()
