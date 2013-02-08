from cluster import *
import numpy as np
import random


def parse_line(line):
    dat = line.split(',')
    return SyntheticDataInstance(dat[0], dat[1], dat[2])


def k_means_objective(clusters):
    sum_of_squares = 0.0

    for c in clusters:
        for x in c.points:
            sum_of_squares += np.inner(x - c.mu, x - c.mu)

    return sum_of_squares


def assign_random_center(cluster_list):
    total_points = []
    K = len(cluster_list)
    [total_points.extend(c.points) for c in cluster_list]
    num_points = len(total_points)

    draw_wo_replacement = random.sample(range(num_points), K)

    for i, c in enumerate(cluster_list):
        c.mu = total_points[draw_wo_replacement[i]]
