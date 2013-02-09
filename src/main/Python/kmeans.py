from operator import itemgetter
from lloyds import assign_random_center
from kmpp import kmpp
from utilities import *
from cluster import *
import random


def reclassify_points(cluster_list):
    # List of the centers
    centers = [c.mu for c in cluster_list]

    # For each cluster
    for i, clust in enumerate(cluster_list):
        # For each point in the cluster
        for j, x in enumerate(clust.points):
            # Calculate the distance from this point to each of
            # the centers
            dist = [np.inner(x - c, x - c) for c in centers]
            # Find index of the smallest distance
            min_ind = min(enumerate(dist), key=itemgetter(1))[0]

            # If this point is not already in this cluster
            if min_ind != i:
                # Reassign this point to the closest cluster
                cluster_list[min_ind].points.append(x)
                # Remove it from the current cluster
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


def fill_up_clusters(cluster_list):
    K = len(cluster_list)
    # Load synthetic data file:
    with open('2DGaussianMixture.csv', mode='r') as f:
        next(f)  # Skip first line
        for line in f:
            r = random.randrange(K)
            d = parse_line(line)
            cluster_list[r].points.append(d.x)


def k_means(K, initialization):
    KClusters = [Cluster() for k in range(K)]

    fill_up_clusters(KClusters)

    if initialization == 'lloyds':
        assign_random_center(KClusters)
    elif initialization == 'km++':
        kmpp(KClusters)
    else:
        print "Problem!"

    kmo_old = 0
    for i in range(50):
        reclassify_points(KClusters)
        find_average_centers(KClusters)
        kmo = k_means_objective(KClusters)
        if abs(kmo - kmo_old) < 0.000001:
            break
        kmo_old = kmo

    print "Iteration: " + str(i)
    print kmo
    print_points_to_file(initialization + '_' + str(K) + \
        '_clusters.csv', KClusters)


def main():
    k_means(20, 'lloyds')
    k_means(20, 'km++')


if __name__ == '__main__':
    main()
