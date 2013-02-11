from utilities import *


EPS = np.finfo(float).eps


def get_data():
    data = np.loadtxt('./2DGaussianMixture.csv', dtype='float', delimiter=',',
     skiprows=1, usecols=(1, 2))
    return data


def objective(X, labels, centers):
    score = 0.0
    for i, x in enumerate(X):
        diff = x - centers[labels[i]]
        score += np.dot(diff, diff)
    return score


class KMeans():
    def __init__(self, n_clusters=1, init='lloyds'):
        self.n_clusters = n_clusters
        self.init = init

    def _initialization_step(self, X):
        centers = np.zeros((self.n_clusters, X.shape[1]))
        if self.init == 'lloyds':
            ind = np.random.randint(0, X.shape[0], self.n_clusters)
            centers = X[ind]
        if self.init == 'km++':
            first_center_id = np.random.randint(0, X.shape[0], 1)
            centers[0] = X[first_center_id]

            X_squared_norms = np.zeros(X.shape[0])
            for i, x in enumerate(X):
                X_squared_norms[i] = np.dot(x, x)

            for c in range(1, self.n_clusters):
                pass
        return centers

    def _expectation_step(self, X, centers):
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

    def _maximization_step(self, X, labels):
        n_features = X.shape[1]
        centers = np.zeros((self.n_clusters, n_features))

        n_samples_in_cluster = np.bincount(labels, minlength=self.n_clusters)

        # Fix this part!
        # Assign a point
        empty_clusters = np.where(n_samples_in_cluster == 0)[0]

        for i in empty_clusters:
            n_samples_in_cluster[i] = 1
        for i, x in enumerate(X):
            centers[labels[i]] += x
        centers /= n_samples_in_cluster[:, np.newaxis]
        return centers

    def fit(self, X):
        kmo = 1.0
        kmo_old = 0.0

        centers = self._initialization_step(X)
        while(abs(kmo - kmo_old) > 0.0001):
            kmo_old = kmo
            labels = self._expectation_step(X, centers)
            centers = self._maximization_step(X, labels)
            kmo = objective(X, labels, centers)
        return centers


class GMM_EM():
    def __init__(self, n_clusters=1, n_iters=100):
        self.n_clusters = n_clusters
        self.n_iters = n_iters
        self.weights = np.ones(self.n_clusters) / self.n_clusters

    def _init_means_covars(self, X):
        self.means = KMeans(self.n_clusters, 'lloyds')._initialization_step(X)
        n_features = X.shape[1]
        self.covars = np.tile(np.eye(n_features), (self.n_clusters, 1, 1))

    def _log_multivariate_normal_density(self, X, means, covars,
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
                # print c
                # print mu
                # print cv
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

        # print log_prob

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
        if X.ndim == 1:
            X = X[:, np.newaxis]
        if X.shape[1] != self.means.shape[1]:
            raise ValueError('the shape of X  is not compatible with self')

        lpr = (self._log_multivariate_normal_density(X, self.means, self.covars)
               + np.log(self.weights))
        logprob = self._logsumexp(lpr, axis=1)
        responsibilities = np.exp(lpr - logprob[:, np.newaxis])
        return logprob, responsibilities

    def _do_mstep(self, X, responsibilities, min_covar=1.e-7):
        """ Perform the Mstep of the EM algorithm and return the class weihgts.
        """
        weights = responsibilities.sum(axis=0)
        weighted_X_sum = np.dot(responsibilities.T, X)
        inverse_weights = 1.0 / (weights[:, np.newaxis] + 10 * EPS)

        self.weights = (weights / (weights.sum() + 10 * EPS) + EPS)

        self.means = weighted_X_sum * inverse_weights

        self.covars = self._covar_mstep(
            X, responsibilities, weighted_X_sum, inverse_weights)

        return weights

    def _covar_mstep(self, X, responsibilities, weighted_X_sum, norm,
                          min_covar=1.e-7):
        """Performing the covariance M step for full cases"""
        # Eq. 12 from K. Murphy, "Fitting a Conditional Linear Gaussian
        # Distribution"
        n_features = X.shape[1]
        cv = np.empty((self.n_clusters, n_features, n_features))
        for c in range(self.n_clusters):
            post = responsibilities[:, c]
            # Underflow Errors in doing post * X.T are  not important
            np.seterr(under='ignore')
            avg_cv = np.dot(post * X.T, X) / (post.sum() + 10 * EPS)
            mu = self.means[c][np.newaxis]
            cv[c] = (avg_cv - np.dot(mu.T, mu) + min_covar * np.eye(n_features))
        return cv

    def fit(self, X):
        self._init_means_covars(X)

        llsum = 1.0
        llsum_old = 0.0
        # for i in range(40):
        for i in range(self.n_iters):
        # while(abs(llsum - llsum_old) > 0.000001):
            llsum_old = llsum
            curr_log_likelihood, responsibilities = self.eval(X)
            llsum = curr_log_likelihood.sum()
            if i > 0 and abs(llsum - llsum_old) < 0.0001:
                break
            self._do_mstep(X, responsibilities)

        labels = responsibilities.argmax(axis=1)
        print objective(X, labels, self.means)

        return labels, responsibilities
        # self.print_solutions(X, responsibilities)

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
    import matplotlib.pyplot as plt
    import math
    from scipy import linalg
    from matplotlib.patches import Ellipse

    X = get_data()
    G = GMM_EM(3)
    l, r = G.fit(X)

    def pie_radius_points(theta1, theta2):
        cx = [0] + np.cos(np.linspace(2.0 * math.pi * theta1,
            2.0 * math.pi * theta2, 10)).tolist()
        cy = [0] + np.sin(np.linspace(2.0 * math.pi * theta1,
            2.0 * math.pi * theta2, 10)).tolist()
        c = list(zip(cx, cy))
        return c

    def plotEllipse(pos, P, edge, face):
        U, s, Vh = linalg.svd(P)
        orient = math.atan2(U[1, 0], U[0, 0]) * 180 / math.pi
        ellipsePlot = Ellipse(xy=pos, width=2.0 * math.sqrt(s[0]),
            height=2.0 * math.sqrt(s[1]), angle=orient,  #facecolor=face,
            edgecolor=edge)
        ax = plt.gca()
        ax.add_patch(ellipsePlot)
        return ellipsePlot

    def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
        """
        Plots an `nstd` sigma error ellipse based on the specified covariance
        matrix (`cov`). Additional keyword arguments are passed on to the
        ellipse patch artist.

        Parameters
        ----------
            cov : The 2x2 covariance matrix to base the ellipse on
            pos : The location of the center of the ellipse. Expects a 2-element
                sequence of [x0, y0].
            nstd : The radius of the ellipse in numbers of standard deviations.
                Defaults to 2 standard deviations.
            ax : The axis that the ellipse will be plotted on. Defaults to the
                current axis.
            Additional keyword arguments are pass on to the ellipse patch.

        Returns
        -------
            A matplotlib ellipse artist
        """
        def eigsorted(cov):
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            return vals[order], vecs[:, order]

        if ax is None:
            ax = plt.gca()

        vals, vecs = eigsorted(cov)
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

        # Width and height are "full" widths, not radius
        width, height = 2 * nstd * np.sqrt(vals)
        ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

        ax.add_artist(ellip)
        return ellip

    size = 15

    fig = plt.figure()

    ax = fig.add_subplot(111)
    for i, p in enumerate(X):
        ang = np.cumsum(r[i, :])
        ax.scatter(p[0], p[1],
            marker=(pie_radius_points(0, ang[0]), 0), s=size, facecolor='blue')
        ax.scatter(p[0], p[1],
            marker=(pie_radius_points(ang[0], ang[1]), 0), s=size, facecolor='green')
        ax.scatter(p[0], p[1],
            marker=(pie_radius_points(ang[1], ang[2]), 0), s=size, facecolor='red')

    for i in range(3):
        # plotEllipse(G.means[i], G.covars[i], 'black', '0')
        plot_cov_ellipse(G.covars[i], G.means[i], nstd=2, ax=None, alpha=0.2)

    plt.show()


if __name__ == '__main__':
    main()
