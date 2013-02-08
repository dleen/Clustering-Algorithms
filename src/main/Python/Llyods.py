from utilities import *
from operator import itemgetter


def reclassify_points(cluster_list):
    centers = [c.mu for c in cluster_list]

    for i, clust in enumerate(cluster_list):
        for j, x in enumerate(clust.points):
            dist = [np.inner(x - c, x - c) for c in centers]
            min_ind = min(enumerate(dist), key=itemgetter(1))[0]

            if min_ind != i:
                cluster_list[min_ind].points.append(x)
                cluster_list[i].points.pop(j)


def find_average_centers(cluster_list):
    new_centers = [single_average_center(c) for c in cluster_list]
    for i, c in enumerate(cluster_list):
        c.mu = new_centers[i]


def single_average_center(cluster):
    n = len(cluster.points)
    sum = 0.0
    for x in cluster.points:
        sum += x

    return sum / n
