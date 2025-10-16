import numpy as np

from utils.load_data import MAX_LENGTH, N_CHANNELS

def format_signal(signal: np.ndarray) -> np.ndarray:
    """
    Removes the padded values from a signal

    Args:
        signal (np.ndarray): a (padded) time series with 12 channels of  
        cepstrum coefficients.

    Returns:
        np.ndarray: the same signal as the input signal, but without the 
        padded values (0's).
    """

    last_row = np.nonzero(signal)[0][-1] + 1        # last index of non-padded row
    signal = np.array([signal[:last_row, channel] for channel in range(N_CHANNELS)])
    return signal


def get_distance_matrix(signal1: np.ndarray, signal2: np.ndarray) -> np.ndarray:
    """
    Generates the distance matrix required for the Dynamic Time Warping (DTW)
    procedure.

    Args:
        signal1 (np.ndarray): first signal used in the distance matrix
        signal2 (np.ndarray): second signal used in the distance matrix.

    Returns:
        np.ndarray: the distance matrix with shape (length signal 1, length 
        signal 2).
    """

    # initialize distance-matrix
    distance_matrix = np.zeros((signal1.shape[0], signal2.shape[0]))

    # loop over every entry in the distance-matrix
    for idx1 in range(distance_matrix.shape[0]):
        for idx2 in range(distance_matrix.shape[1]):
            # local distance is the squared distance between one point in signal 1
            # and one point in signal 2
            local_distance = (signal1[idx1] - signal2[idx2]) ** 2
            
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
    """
    Calculates the distance according to the Dynamic Time Warping (DTW)
    procedure, given a distance matrix.

    Args:
        distance_matrix (np.ndarray): a distance matrix used for the DTW
        procedure.

    Returns:
        float: the accumulated distance from following the path on the
        distance matrix, according to the DTW procedure.
    """

    distance = 0
    # start at the last row, last column index of the distance matrix
    idx = (distance_matrix.shape[0] - 1, distance_matrix.shape[1] - 1)

    # while not at the first row, first column index of the distance matrix
    while idx != (0, 0):
        distance += distance_matrix[idx[0], idx[1]]

        # select the possible indexes to move to and the values of these indexes
        # in the distance matrix
        connections = []
        idxs = []
        if idx[0] > 0:
            idxs.append((idx[0] - 1, idx[1]))
            connections.append(distance_matrix[idx[0] - 1, idx[1]])
        if idx[1] > 0:
            idxs.append((idx[0], idx[1] - 1))
            connections.append(distance_matrix[idx[0], idx[1] - 1])
        if idx[0] > 0 and idx[1] > 0:
            idxs.append((idx[0] - 1, idx[1] - 1))
            connections.append(distance_matrix[idx[0] - 1, idx[1] - 1])

        # select the index with the lowest corresponding value
        idx = idxs[np.argmin(np.array(connections))]

    return distance


def DTW(signal1: np.ndarray, signal2: np.ndarray) -> float:
    """
    Calculates the summed Dynamic Time Warping (DTW) distances over all
    channels in the given signals.

    Args:
        signal1 (np.ndarray): first signal used to compare DTW distance with.
        signal2 (np.ndarray): second signal used to compare DTW distance with.

    Returns:
        float: summed DTW distances over all 12 channels in the signals.
    """

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

    # strip the padded values from the signals
    signal1 = format_signal(signal1)
    signal2 = format_signal(signal2)

    # sum the DTW distances over all channels
    total_distance = 0
    for channel in range(N_CHANNELS):
        distance_matrix = get_distance_matrix(signal1[channel], signal2[channel])
        total_distance += get_distance(distance_matrix)

    return total_distance
