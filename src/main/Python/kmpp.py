def pick_first_center(cluster_list):
    total_points = []
    [total_points.extend(c.points) for c in cluster_list]
    num_points = len(total_points)

    r = random.randrange(num_points)

    cluster_list[0].mu = total_points[r]




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

    pick_first_center(KClusters)




if __name__ == '__main__':
    main()
