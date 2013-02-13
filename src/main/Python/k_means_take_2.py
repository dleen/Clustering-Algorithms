import numpy as np


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
    def __init__(self, n_clusters=3, init='lloyds'):
        self.n_clusters = n_clusters
        self.init = init

    def _initialization_step(self, X):
        centers = np.zeros((self.n_clusters, X.shape[1]))
        if self.init == 'lloyds':
            ind = np.random.randint(0, X.shape[0], self.n_clusters)
            centers = X[ind]
        elif self.init == 'km++':
            centers = self._init_kmpp(X, centers)

        return centers

    def _init_kmpp(self, X, centers):
        first_center_id = np.random.randint(0, X.shape[0], 1)
        centers[0] = X[first_center_id]

        X_squared_norms = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            X_squared_norms[i] = np.dot(x, x)

        for c in range(1, self.n_clusters):
            closest_dist_sq = self._distances_to_centers(centers[0:(c)], X,
                Y_norm_sq=X_squared_norms, sq=True)
            closest_dist_sq = closest_dist_sq.min(axis=0)
            probs = closest_dist_sq / sum(closest_dist_sq)
            ind = np.random.choice(range(X.shape[0]), p=probs)
            centers[c] = X[ind]
        return centers

    def _distances_to_centers(self, X, Y, Y_norm_sq=None, sq=False):
        if X.ndim == 1:
            XX = np.sum(X * X)
        else:
            XX = np.sum(X * X, axis=1)[:, np.newaxis]
        distances = np.dot(X, Y.T)
        distances *= -2
        distances += XX
        distances += Y_norm_sq
        return distances if sq else np.sqrt(distances)

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
