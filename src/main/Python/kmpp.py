import random
import numpy as np


def pick_first_center(cluster_list, total_points):
    num_points = len(total_points)
    r = random.randrange(num_points)

    cluster_list[0].mu = total_points[r]


def compute_probability_list(cluster_list, total_points, iter):
    centers = [c.mu for c in cluster_list[0:iter]]

    Dsq = []
    for x in total_points:
        distsq = [np.inner(x - c, x - c) for c in centers]
        Dsq.append(min(distsq))

    return Dsq / sum(Dsq)


def pick_cluster_center_at_random(total_points, probability_list):
    r = random.random()
    i = 0
    n = len(probability_list)

    while(r >= 0 and i < n):
        r -= probability_list[i]
        i += 1

    return total_points[i - 1]


def kmpp(cluster_list):
    total_points = []
    K = len(cluster_list)

    [total_points.extend(c.points) for c in cluster_list]

    pick_first_center(cluster_list, total_points)

    for j in range(1, K):
        probs = compute_probability_list(cluster_list, total_points, j)
        cluster_list[j].mu = pick_cluster_center_at_random(total_points, probs)
