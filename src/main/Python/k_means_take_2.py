import numpy as np


def get_data():
    data = np.loadtxt('./2DGaussianMixture.csv', dtype='float', delimiter=',',
     skiprows=1, usecols=(1, 2))
    return data


def objective(X, labels, centers):
    score = 0.0
    # for i, x in enumerate(X):
    #     diff = x - centers[labels[i]]
    #     score += np.dot(diff, diff)

    dd = X - centers[labels]

    # for i in range(dd.shape[0]):
    #     for j in range(dd.shape[1]):
    #         score += dd[i, j] * dd[i, j]
    # print dd.shape
    # a = dd * dd
    # print a.shape
    DD = np.sum(np.multiply(dd, dd))
    return DD
    # dd = X -
    # XX = np.sum(X * X, axis=1)
    # return score


class KMeans():
    def __init__(self, n_clusters=3, init='lloyds', user_centers=None):
        self.n_clusters = n_clusters
        self.init = init
        self.user_centers = user_centers

    def _initialization_step(self, X):
        centers = np.zeros((self.n_clusters, X.shape[1]))
        if self.init == 'lloyds':
            ind = np.random.randint(0, X.shape[0], self.n_clusters)
            centers = X[ind]
        elif self.init == 'km++':
            centers = self._init_kmpp(X, centers)
        elif self.init == 'user':
            centers = self.user_centers
        return centers

    def _init_kmpp(self, X, centers):
        first_center_id = np.random.randint(0, X.shape[0], 1)
        centers[0] = X[first_center_id]

        # X_squared_norms = np.zeros(X.shape[0])
        # for i, x in enumerate(X):
        #     X_squared_norms[i] = np.dot(x, x)
        X_squared_norms = np.sum(np.multiply(X, X), axis=1)
        # X_squared_norms = (X ** 2).sum(axis=1)

        # X_squared_norms = X_squared_norms[np.newaxis, :]
        # print Y.shape
        # print X_squared_norms.shape

        for c in range(1, self.n_clusters):
            closest_dist_sq = self._distances_to_centers(centers[0:c], X,
                X_squared_norms, squared=True)
            closest_dist_sq = closest_dist_sq.min(axis=0)
            if closest_dist_sq.ndim > 1:
                closest_dist_sq = np.ravel(closest_dist_sq)
            probs = closest_dist_sq / sum(closest_dist_sq)
            ind = np.random.choice(range(X.shape[0]), p=probs)
            centers[c] = X[ind]
        return centers

    def _distances_to_centers(self, X, Y, Y_norm_squared=None, squared=False):
        XX = np.sum(X * X, axis=1)[:, np.newaxis]
        YY = Y_norm_squared
        if YY.ndim > 1:
            YY = np.rollaxis(YY, axis=1)
        distances = np.dot(X, Y.T)
        distances *= -2
        distances += XX
        distances += YY

        return distances if squared else np.sqrt(distances)

    def _expectation_step(self, X, centers):
        n_samples = X.shape[0]
        labels = - np.ones(n_samples, dtype=np.int)
        centers_squared_norms = np.sum(np.multiply(centers, centers), axis=1)
        X_squared_norms = np.sum(np.multiply(X, X), axis=1)

        for sample_ind, sample in enumerate(X):
            min_dist = -1
            for center_ind, center in enumerate(centers):
                dist = 0.0
                if sample.ndim > 1:
                    dist += -2 * np.sum(np.multiply(sample, center), axis=1)
                else:
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
        for i in range(X.shape[0]):
            centers[labels[i]] += np.ravel(X[i])
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

        self.labels = labels
        self.centers = centers
        return centers


def main():
    X = get_data()
    K1 = KMeans(3, 'lloyds')
    K2 = KMeans(3, 'km++')
    K1.fit(X)
    K2.fit(X)

    print objective(X, K1.labels, K1.centers)
    print objective(X, K2.labels, K2.centers)


if __name__ == '__main__':
    main()
