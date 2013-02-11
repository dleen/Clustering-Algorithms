import numpy as np
from kmpp import kmpp
from utilities import *
from kmeans import fill_up_clusters
from cluster import *


def _log_multivariate_normal_density_full(X, means, covars):
    """Log probability for full covariance matrices.
    """
    from scipy import linalg
    if hasattr(linalg, 'solve_triangular'):
        # only in scipy since 0.9
        solve_triangular = linalg.solve_triangular
    else:
        # slower, but works
        solve_triangular = linalg.solve
    n_samples = 1
    n_dim = 2
    nmix = len(means)
    log_prob = np.empty((n_samples, nmix))

    cv_chol = linalg.cholesky(covars, lower=True)

    cv_log_det = 2 * np.sum(np.log(np.diagonal(cv_chol)))
    cv_sol = solve_triangular(cv_chol, (X - means).T, lower=True).T

    log_prob = - .5 * (np.sum(cv_sol ** 2) +
                             n_dim * np.log(2 * np.pi) + cv_log_det)

    return log_prob


def log_prob_x_cond_theta(x, cluster):
    # print x
    # print x.shape
    # print cluster.cov
    logp = _log_multivariate_normal_density_full(x, cluster.mu,
        cluster.cov)

    return logp + np.log(cluster.weight)


def responsibility(x, lpr, cluster_list):
    arr = []
    for c in cluster_list:
        arr.append(log_prob_x_cond_theta(x, c))

    logprob = logsumexp(arr)

    return np.exp(lpr - logprob)


def logsumexp(arr):
    # Use the max to normalize, as with the log this is what accumulates
    # the less errors
    vmax = max(arr)
    out = np.log(np.sum(np.exp(arr - vmax)))
    out += vmax
    return out


def calculate_normalization_factor(x, cluster_list):
    norm = 0.0
    for c in cluster_list:
        norm += c.weight * prob_x_cond_theta(x, c)

    # print "norm",
    # print norm

    return norm


def em_step(cluster_list):
    r = []

    for c in cluster_list:
        # print c.cov
        for x in c.points:
            lpr = log_prob_x_cond_theta(x, c)
            rtemp = responsibility(x, lpr, cluster_list)
            # print "rtemp",
            # print rtemp
            r.append(rtemp)
            # if np.isnan(r).any():
            #     break
        # print len(r)
        # print sum(r)
        c.weight = calc_new_weight(r)
        # print c.weight
        calc_new_parameters(r, c)
        r = []


def calc_new_weight(r):
    return sum(r) / len(r)


def calc_new_parameters(r, cluster):
    rk = sum(r)
    new_mu = 0.0
    new_cov = np.empty((2, 2))

    for i, x in enumerate(cluster.points):
        new_mu += r[i] * x
        new_cov += r[i] * np.outer(x, x)

    # print new_cov

    cluster.cov = new_cov / rk - np.outer(cluster.mu, cluster.mu)
    # print cluster.cov
    cluster.mu = new_mu / rk


def initialize_weights(cluster_list):
    K = len(cluster_list)
    for c in cluster_list:
        c.weight = 1.0 / K


def main(K, initialization):
    # KClusters = [Cluster() for k in range(K)]

    b = np.loadtxt('2DGaussianMixture.csv', dtype='float', delimiter=',',
        skiprows=1)

    print b.shape

    # fill_up_clusters(KClusters)
    # kmpp(KClusters)
    # initialize_weights(KClusters)

    # kmo_old = 0
    # for i in range(50):
    #     em_step(KClusters)
    #     kmo = k_means_objective(KClusters)
    #     if abs(kmo - kmo_old) < 0.000001:
    #         break
    #     kmo_old = kmo

    # print "Iteration: " + str(i)
    # print kmo
    # print_points_to_file(initialization + '_' + str(K) + \
    #     '_clusters.csv', KClusters)


if __name__ == '__main__':
    main(2, 'gmm_em')
