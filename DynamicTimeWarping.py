import numpy as np

from loadData import MAX_LENGTH

N_CHANNELS = 12

def format_signal(signal: np.ndarray) -> np.ndarray:
    last_row = np.nonzero(signal)[0][-1] + 1        # last index of non-padded row
    signal = np.array([signal[:last_row, channel] for channel in range(N_CHANNELS)])
    return signal


def get_distance_matrix(signal1: np.ndarray, signal2: np.ndarray) -> np.ndarray:
    # initialize distance-matrix
    distance_matrix = np.zeros((signal1.shape[0], signal2.shape[0]))

    # loop over every entry in the distance-matrix
    for idx1 in range(distance_matrix.shape[0]):
        for idx2 in range(distance_matrix.shape[1]):
            # local distance is the eucledian distance between one point in signal 1
            # and one point in signal 2
            local_distance = abs(signal1[idx1] - signal2[idx2])
            
            # lowest connection is the smallest entry 1 row and/or column lower than
            # the current point
            lowest_connection = 0
            if idx1 > 0 and idx2 > 0:
                lowest_connection = min(distance_matrix[idx1 - 1, idx2], distance_matrix[idx1, idx2 - 1], distance_matrix[idx1 - 1, idx2 - 1])
            elif idx1 > 0:
                lowest_connection = distance_matrix[idx1 - 1, idx2]
            elif idx2 > 0:
                lowest_connection = distance_matrix[idx1, idx2 - 1]

            distance_matrix[idx1, idx2] = local_distance + lowest_connection

    return distance_matrix

def get_distance(distance_matrix: np.ndarray) -> float:
    distance = 0
    idx = (distance_matrix.shape[0] - 1, distance_matrix.shape[1] - 1)

    while idx != (0, 0):
        distance += distance_matrix[idx[0], idx[1]]

        connections = np.array([])
        idxs = []
        if idx[0] > 0:
            idxs.append((idx[0] - 1, idx[1]))
            connections = np.append(connections, distance_matrix[idx[0] - 1, idx[1]])
        if idx[1] > 0:
            idxs.append((idx[0], idx[1] - 1))
            connections = np.append(connections, distance_matrix[idx[0], idx[1] - 1])
        if idx[0] > 0 and idx[1] > 0:
            idxs.append((idx[0] - 1, idx[1] - 1))
            connections = np.append(connections, distance_matrix[idx[0] - 1, idx[1] - 1])
        
        idx = idxs[np.argmin(connections)]

    return distance


# Dynamic Time Warp function, meant as a measure of distance between
# speach signals. Custom to our dataset
def DTW(signal1: np.ndarray, signal2: np.ndarray) -> float:
    if signal1.shape != (MAX_LENGTH, N_CHANNELS):
        print(f"expected second signal to have {MAX_LENGTH} time steps and \
              {N_CHANNELS} channels, got {signal1.shape[0]} time steps and \
                {signal1.shape[1]} channels instead.")
        return

    if signal1.shape != (MAX_LENGTH, N_CHANNELS):
        print(f"expected second signal to have {MAX_LENGTH} time steps and \
              {N_CHANNELS} channels, got {signal2.shape[0]} time steps and \
                {signal2.shape[1]} channels instead.")
        return
    
    signal1 = format_signal(signal1)
    signal2 = format_signal(signal2)

    total_distance = 0
    for channel in range(N_CHANNELS):
        distance_matrix = get_distance_matrix(signal1[channel], signal2[channel])
        total_distance += get_distance(distance_matrix)

    return total_distance
