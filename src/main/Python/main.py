from cluster import *
from utilities import *
from Llyods import *

import random


def main():
    # Load synthetic data file:
    synthetic_data_file = open('2DGaussianMixture.csv', mode='r')
    next(synthetic_data_file)  # Skip first line

    K = 3
    KClusters = [Cluster() for k in range(K)]

    for line in synthetic_data_file:
        r = random.randrange(K)
        d = parse_line(line)
        KClusters[r].points.append(d.x)

    synthetic_data_file.close()

    assign_random_center(KClusters)
    print k_means_objective(KClusters)

    for i in range(20):
        print "Iteration: " + str(i)
        reclassify_points(KClusters)
        print k_means_objective(KClusters)
        find_average_centers(KClusters)
        print k_means_objective(KClusters)


if __name__ == '__main__':
    main()
