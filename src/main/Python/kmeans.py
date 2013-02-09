from operator import itemgetter
from lloyds import assign_random_center
from utilities import *
from cluster import *
import random


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


def k_means(K, initialization):
    KClusters = [Cluster() for k in range(K)]

    # Load synthetic data file:
    with open('2DGaussianMixture.csv', mode='r') as f:
        next(f)
        for line in f:
            r = random.randrange(K)
            d = parse_line(line)
            KClusters[r].points.append(d.x)

    # synthetic_data_file = open('2DGaussianMixture.csv', mode='r')
    # next(synthetic_data_file)  # Skip first line

    # KClusters = [Cluster() for k in range(K)]

    # for line in synthetic_data_file:
    #     r = random.randrange(K)
    #     d = parse_line(line)
    #     KClusters[r].points.append(d.x)

    # synthetic_data_file.close()

    if initialization == 'lloyds':
        assign_random_center(KClusters)

    kmo_old = 0
    for i in range(25):
        print "Iteration: " + str(i)
        reclassify_points(KClusters)
        find_average_centers(KClusters)
        kmo = k_means_objective(KClusters)
        if abs(kmo - kmo_old) < 0.01:
            break
        kmo_old = kmo

    print kmo
    print_points_to_file('lloyds_' + str(K) + '_clusters.csv', KClusters)


def main():
    k_means(3, 'lloyds')


if __name__ == '__main__':
    main()
