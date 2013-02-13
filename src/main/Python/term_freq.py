from scipy.io import mmread
import numpy as np


def get_bbc_data():
    # term-id, doc-id, freq
    X = mmread('bbc_data/bbc.mtx')
    X = X.tocsr()
    return X


def get_class_data():
    data = np.loadtxt('bbc_data/bbc.classes', dtype='int', delimiter=' ',
        usecols=[1])
    return data


def get_centers_data():
    data = np.loadtxt('bbc_data/bbc.centers')
    return data


def calc_idf(X):
    D = 1791
    idf = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        idf[i] = X[i].size
        idf[i] = np.log(D / idf[i])
    return idf


def tf_idf(X):
    idf = calc_idf(X)
    for i in range(X.shape[0]):
        for j in X[i].indices:
            X[i, j] = idf[i] * X[i, j]


def avg_tfidf(t, C, X):
    pass


def main():
    X = get_bbc_data()
    C = get_class_data()
    centers = get_centers_data()

    test = np.where(C == 1)
    print test

    tf_idf(X)


if __name__ == '__main__':
    main()
