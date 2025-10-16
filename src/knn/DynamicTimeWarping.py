import numpy as np

from src.utils.load_data import MAX_LENGTH, N_CHANNELS
from src.knn.knn_pipeline import remove_padding


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
    signal1 = remove_padding(signal1)
    signal2 = remove_padding(signal2)

    # sum the DTW distances over all channels
    total_distance = 0
    for channel in range(N_CHANNELS):
        distance_matrix = get_distance_matrix(signal1[channel], signal2[channel])
        total_distance += np.sqrt(distance_matrix[-1, -1])
    return total_distance