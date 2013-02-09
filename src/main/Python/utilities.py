from cluster import SyntheticDataInstance
import numpy as np


def parse_line(line):
    dat = line.split(',')
    return SyntheticDataInstance(dat[0], dat[1], dat[2])


def k_means_objective(clusters):
    sum_of_squares = 0.0

    for c in clusters:
        for x in c.points:
            sum_of_squares += np.inner(x - c.mu, x - c.mu)

    return sum_of_squares


def print_points_to_file(filename, cluster_list):
    with open(filename, 'w') as f:
        f.write('cluster,x1,x2,mu1,mu2\n')
        for i, c in enumerate(cluster_list):
            for x in c.points:
                f.write(str(i) + ',' + str(x[0]) + ',' + \
                     str(x[1]) + ',' + str(c.mu[0]) + \
                     ',' + str(c.mu[1]) + '\n')
