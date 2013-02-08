import numpy as np


class Cluster:
    """ Defines a cluster """
    def __init__(self, mu=np.array([0.0, 0.0]), points=None):
        self.mu = mu
        self.points = points or []

    def __repr__(self):
        return "Cluster with mean: " + str(self.cluster_center)


class SyntheticDataInstance:
    """ A line of Synthetic data """
    def __init__(self, class_num, x1, x2):
        self.class_num = int(class_num)
        self.x = np.array([float(x1), float(x2)])

    def __repr__(self):
        return "Data: " + str(self.class_num) + ', ' + \
            str(self.x[0]) + ', ' + str(self.x[1])
