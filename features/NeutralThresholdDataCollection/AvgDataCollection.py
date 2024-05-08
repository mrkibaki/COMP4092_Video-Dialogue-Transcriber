

def EEBAvgData(distances_left, distances_right, EB_dist):
    # Calculate the average of the collected distances
    average_distance_left = sum(distances_left) / len(distances_left) if distances_left else 0
    average_distance_right = sum(distances_right) / len(distances_right) if distances_right else 0
    average_EB_distance = sum(EB_dist) / len(EB_dist) if EB_dist else 0

    AvgDataSet = {'left': average_distance_left, 'right': average_distance_right, 'ebdist': average_EB_distance}
    return AvgDataSet


def NBAvgData(seg1, seg2, seg3, NB):
    avg_seg1 = sum(seg1)/len(seg1)if seg1 else 0
    avg_seg2 = sum(seg2)/len(seg2)if seg2 else 0
    avg_seg3 = sum(seg3)/len(seg3)if seg3 else 0
    avg_NB = sum(NB)/len(NB)if NB else 0

    NBDataSet = {'seg1': avg_seg1, 'seg2': avg_seg2, 'seg3': avg_seg3, 'NB': avg_NB}
    return NBDataSet
