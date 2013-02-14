from scipy.io import mmread
import numpy as np
from k_means_take_2 import KMeans
from gmm_em_take_2 import GMM_EM


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


def get_terms_data():
    data = np.loadtxt('bbc_data/bbc.terms', dtype='string')
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


def avg_tfidf(C, X):
    avg_c = np.zeros((5, 99))
    for i in range(5):
        clist = np.where(C == i)[0]
        for j in range(X.shape[0]):
            for k in clist:
                avg_c[i, j] += X[j, k]
            avg_c[i, j] /= clist.size
    return avg_c


def top_5(C, X):
    terms = get_terms_data()
    class_names = ['Business', 'Entertainment',
        'Politics', 'Sport', 'Tech']

    avg = avg_tfidf(C, X)
    sorted_ind = avg.argsort()

    for i in range(5):
        print class_names[i]
        print terms[sorted_ind[i, :-6:-1]]
        print avg[i, sorted_ind[i, :-6:-1]]


def main():
    X = get_bbc_data()
    C = get_class_data()
    centers = get_centers_data()

    tf_idf(X)
    # top_5(C, X)

    # from sklearn import mixture
    # g = mixture.GMM(n_components=5)
    # g.fit(X.T.todense())

    # K = GMM_EM(n_clusters=5, n_iters=100, init='user', user_centers=centers)
    K = KMeans(n_clusters=3, init='km++')
    K.fit(X.T.todense())


if __name__ == '__main__':
    main()
