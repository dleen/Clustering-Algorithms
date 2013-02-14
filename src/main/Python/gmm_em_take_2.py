import numpy as np
from k_means_take_2 import *


EPS = np.finfo(float).eps


class GMM_EM():
    def __init__(self, n_clusters=1, n_iters=100, init='km++',
            user_centers=None):
        self.n_clusters = n_clusters
        self.n_iters = n_iters
        self.init = init
        self.weights = np.ones(self.n_clusters) / self.n_clusters
        self.user_centers = user_centers
        self.min_covar = 1.e-3

    def _init_means_covars(self, X):
        if self.init == 'user':
            self.means = self.user_centers
        else:
            self.means = KMeans(self.n_clusters,
                self.init)._initialization_step(X)
        n_features = X.shape[1]
        # self.covars = np.tile(0.01 * np.eye(n_features), (self.n_clusters, 1, 1))
        self.covars = np.tile(np.cov(X.T) + self.min_covar * np.eye(X.shape[1]),
            (self.n_clusters, 1, 1))

    def _log_multivariate_normal_density(self, X, means, covars,
        min_covar=1.e-3):
        from scipy import linalg
        solve_triangular = linalg.solve_triangular

        n_samples, n_dim = X.shape
        nmix = len(means)
        log_prob = np.empty((n_samples, nmix))
        for c, (mu, cv) in enumerate(zip(means, covars)):
            cv_chol = linalg.cholesky(cv, lower=True)
            cv_log_det = 2 * np.sum(np.log(np.diagonal(cv_chol)))
            cv_sol = solve_triangular(cv_chol, (X - mu).T, lower=True).T
            log_prob[:, c] = - .5 * (np.sum(cv_sol ** 2, axis=1) +
                                     n_dim * np.log(2 * np.pi) + cv_log_det)
        return log_prob

    def _logsumexp(self, arr, axis=0):
        """Computes the sum of arr assuming arr is in the log domain.

        Returns log(sum(exp(arr))) while minimizing the possibility of
        over/underflow.

        Examples
        --------

        >>> import numpy as np
        >>> from sklearn.utils.extmath import logsumexp
        >>> a = np.arange(10)
        >>> np.log(np.sum(np.exp(a)))
        9.4586297444267107
        >>> logsumexp(a)
        9.4586297444267107
        """
        arr = np.rollaxis(arr, axis)
        # Use the max to normalize, as with the log this is what accumulates
        # the less errors
        vmax = arr.max(axis=0)
        out = np.log(np.sum(np.exp(arr - vmax), axis=0))
        out += vmax
        return out

    def eval(self, X):
        """Evaluate the model on data

        Compute the log probability of X under the model and
        return the posterior distribution (responsibilities) of each
        mixture component for each element of X.

        Parameters
        ----------
        X: array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        Returns
        -------
        logprob: array_like, shape (n_samples,)
            Log probabilities of each data point in X
        responsibilities: array_like, shape (n_samples, n_components)
            Posterior probabilities of each mixture component for each
            observation
        """

        X = np.asarray(X)

        lpr = (self._log_multivariate_normal_density(X, self.means, self.covars)
               + np.log(self.weights))
        logprob = self._logsumexp(lpr, axis=1)
        responsibilities = np.exp(lpr - logprob[:, np.newaxis])
        return logprob, responsibilities

    def _maximization_step(self, X, responsibilities,
            min_covar=1.e-3):
        lam = 0.2
        weights = responsibilities.sum(axis=0)
        weighted_X_sum = np.dot(responsibilities.T, X)
        inverse_weights = 1.0 / (weights[:, np.newaxis] + 10 * EPS)

        self.weights = (weights / (weights.sum() + 10 * EPS) + EPS)

        self.means = np.multiply(weighted_X_sum, inverse_weights)

        self.covars = self._covar_mstep(
            X, responsibilities, weighted_X_sum, inverse_weights, lam=lam)
        return weights
        # weights = responsibilities.sum(axis=0)
        # weighted_X_sum = np.dot(responsibilities.T, X)
        # inverse_weights = 1.0 / (weights[:, np.newaxis] + 10 * EPS)

        # self.weights_ = (weights / (weights.sum() + 10 * EPS) + EPS)
        # self.means_ = weighted_X_sum * inverse_weights
        # covar_mstep_func = _covar_mstep_funcs[self.covariance_type]
        # self.covars_ = covar_mstep_func(
        #     self, X, responsibilities, weighted_X_sum, inverse_weights,
        #     min_covar)
        # return weights

    def _covar_mstep(self, X, responsibilities, weighted_X_sum, norm,
                          min_covar=1.e-3, lam=0):
        n_features = X.shape[1]
        cv = np.empty((self.n_clusters, n_features, n_features))
        for c in range(self.n_clusters):
            post = responsibilities[:, c]
            # avg_cv = np.dot(post * X.T, X) / (post.sum() + 10 * EPS)
            avg_cv = np.dot(np.multiply(post, X.T), X) / (post.sum() + 10 * EPS)
            mu = self.means[c][np.newaxis]
            cv[c] = (avg_cv - np.dot(mu.T, mu) + min_covar * np.eye(n_features))
            cv[c] *= (1 - lam)
            cv[c] += lam * np.eye(n_features)
        return cv

    def fit(self, X):
        self._init_means_covars(X)

        llsum = 1.0
        llsum_old = 0.0

        for i in range(self.n_iters):
            print i
            llsum_old = llsum
            curr_log_likelihood, responsibilities = self.eval(X)
            llsum = curr_log_likelihood.sum()
            if i > 0 and abs(llsum - llsum_old) < 0.0001:
                break
            self._maximization_step(X, responsibilities, min_covar=1.e-3)

        labels = responsibilities.argmax(axis=1)
        print objective(X, labels, self.means)
        return labels, responsibilities

    def print_solutions(self, X, r):
        labels = r.argmax(axis=1)
        with open('gmm_em.csv', 'w') as f:
            f.write('x1,x2,mu1,mu2,r1,r2,r3,cluster\n')
            for i, x in enumerate(X):
                j = labels[i]
                mu = self.means
                f.write(str(x[0]) + ',' + str(x[1]) + \
                ',' + str(mu[j, 0]) + ',' + str(mu[j, 1]) + \
                ',' + str(r[i, 0]) + ',' + str(r[i, 1]) + ',' + str(r[i, 2]) + \
                ',' + str(j) + '\n')


def main():
    X = get_data()
    print X.shape
    G = GMM_EM(n_clusters=3, init='km++')
    l, r = G.fit(X)

    # from plotting import plot_figure
    # plot_figure(X, G, r)

if __name__ == '__main__':
    main()
