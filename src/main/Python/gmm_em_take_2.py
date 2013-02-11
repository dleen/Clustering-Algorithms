from utilities import *


def get_data():
    data = np.loadtxt('2DGaussianMixture.csv', dtype='float', delimiter=',',
     skiprows=1, usecols=(1, 2))
    return data


class KMeans():
    def initialization_step(self, X, n_clusters, init):
        centers = np.zeros((n_clusters, X.shape[1]))
        if init == 'lloyds':
            ind = np.random.randint(0, X.shape[0], n_clusters)
            centers = X[ind]
        if init == 'km++':
            first_center_id = np.random.randint(0, X.shape[0], 1)
            centers[0] = X[first_center_id]

            X_squared_norms = np.zeros(X.shape[0])
            for i, x in enumerate(X):
                X_squared_norms[i] = np.dot(x, x)

            for c in range(1, n_clusters):
                pass

    def expectation_step(self, X, centers, n_clusters):
        n_samples = X.shape[0]
        labels = - np.ones(n_samples, dtype=np.int)

        centers_squared_norms = np.zeros(centers.shape[0])
        X_squared_norms = np.zeros(X.shape[0])
        for i, center in enumerate(centers):
            centers_squared_norms[i] = np.dot(center, center)
        for i, x in enumerate(X):
            X_squared_norms[i] = np.dot(x, x)

        for sample_ind, sample in enumerate(X):
            min_dist = -1
            for center_ind, center in enumerate(centers):
                dist = 0.0
                dist += -2 * np.dot(sample, center)
                dist += centers_squared_norms[center_ind]
                dist += X_squared_norms[sample_ind]

                if min_dist == -1 or dist < min_dist:
                    min_dist = dist
                    labels[sample_ind] = center_ind

        return labels

    def maximization_step(self, X, labels, n_clusters):
        n_features = X.shape[1]
        centers = np.zeros((n_clusters, n_features))

        n_samples_in_cluster = np.bincount(labels, minlength=n_clusters)

        # Fix this part!
        # Assign a point
        empty_clusters = np.where(n_samples_in_cluster == 0)[0]

        for i in empty_clusters:
            n_samples_in_cluster[i] = 1
        for i, x in enumerate(X):
            centers[labels[i]] += x
        centers /= n_samples_in_cluster[:, np.newaxis]
        return centers

    def objective(self, X, labels, centers):
        score = 0.0
        for i, x in enumerate(X):
            diff = x - centers[labels[i]]
            score += np.dot(diff, diff)
        return score


class GMM_EM():
    def __init__(self, n_clusters=1):
        self.n_clusters = n_clusters
        self.weights = np.ones(self.n_clusters) / self.n_clusters

    def _init_means_cov(self, X):
        self.covar = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.means = KMeans(n_clusters, X)

    def _log_multivariate_normal_density_full(self, X, means, covars,
        min_covar=1.e-7):
        """Log probability for full covariance matrices.
        """
        from scipy import linalg
        if hasattr(linalg, 'solve_triangular'):
            # only in scipy since 0.9
            solve_triangular = linalg.solve_triangular
        else:
            # slower, but works
            solve_triangular = linalg.solve
        n_samples, n_dim = X.shape
        nmix = len(means)
        log_prob = np.empty((n_samples, nmix))
        for c, (mu, cv) in enumerate(zip(means, covars)):
            try:
                cv_chol = linalg.cholesky(cv, lower=True)
            except linalg.LinAlgError:
                # The model is most probabily stuck in a component with too
                # few observations, we need to reinitialize this components
                cv_chol = linalg.cholesky(cv + min_covar * np.eye(n_dim),
                                          lower=True)
            cv_log_det = 2 * np.sum(np.log(np.diagonal(cv_chol)))
            cv_sol = solve_triangular(cv_chol, (X - mu).T, lower=True).T
            log_prob[:, c] = - .5 * (np.sum(cv_sol ** 2, axis=1) +
                                     n_dim * np.log(2 * np.pi) + cv_log_det)

        return log_prob

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
        if X.ndim == 1:
            X = X[:, np.newaxis]
        if X.size == 0:
            return np.array([]), np.empty((0, self.n_components))
        if X.shape[1] != self.means_.shape[1]:
            raise ValueError('the shape of X  is not compatible with self')

        lpr = (log_multivariate_normal_density(X, self.means_, self.covars_,
                                               self.covariance_type)
               + np.log(self.weights_))
        logprob = logsumexp(lpr, axis=1)
        responsibilities = np.exp(lpr - logprob[:, np.newaxis])
        return logprob, responsibilities


def main():
    X = get_data()
    # G = GMM_EM()
    K = 3

    kmo = 1.0
    kmo_old = 0.0

    centers = KMeans().initialization_step(X, K, 'lloyds')
    while(abs(kmo - kmo_old) > 0.0001):
        kmo_old = kmo
        labels = KMeans().expectation_step(X, centers, K)
        centers = KMeans().maximization_step(X, labels, K)
        kmo = KMeans().objective(X, labels, centers)
        print kmo


    # G._log_multivariate_normal_density_full()


if __name__ == '__main__':
    main()
