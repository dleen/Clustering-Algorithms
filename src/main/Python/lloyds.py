import random


def assign_random_center(cluster_list):
    total_points = []
    K = len(cluster_list)
    [total_points.extend(c.points) for c in cluster_list]
    num_points = len(total_points)

    draw_wo_replacement = random.sample(range(num_points), K)

    for i, c in enumerate(cluster_list):
        c.mu = total_points[draw_wo_replacement[i]]
