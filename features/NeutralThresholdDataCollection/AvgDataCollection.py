

def AvgData(distances_left, distances_right, EB_dist):
    # Calculate the average of the collected distances
    average_distance_left = sum(distances_left) / len(distances_left) if distances_left else 0
    average_distance_right = sum(distances_right) / len(distances_right) if distances_right else 0
    average_EB_distance = sum(EB_dist) / len(EB_dist) if EB_dist else 0

    AvgDataSet = {'left': average_distance_left, 'right': average_distance_right, 'ebdist': average_EB_distance}
    return AvgDataSet

