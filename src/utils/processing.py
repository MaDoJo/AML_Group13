import numpy as np

from src.utils.load_data import MAX_LENGTH, N_CHANNELS


def get_time_steps(data: np.ndarray) -> np.ndarray:
    """
    Generates a list of how many times each time step is used in the given data.

    Args:
        data (np.ndarray): list of data points. Each data point is a time series
        with 12 channels of cepstrum coefficients.

    Returns:
        np.ndarray: a list with how often each of the MAX_LENGTH time steps
        are used in the given data.
    """

    time_steps = np.zeros(MAX_LENGTH)

    for data_point in data:
        # list of length max-time-steps lenght with 1's for non-padded time steps
        # and 0's for all-0-valued (padded) time steps
        mask = [1 if np.count_nonzero(time_step) > 0 else 0 for time_step in data_point]
        time_steps += mask

    return time_steps


def get_pattern_mean(data: np.ndarray) -> np.ndarray:
    """
    Calculates the mean data point (time series) of all data points in data. Because
    not every time series has the same length, the average of each time step is only
    taken over the non-padded instances of that time step. The data points consist of
    12 individual time series (channels), so the resulting mean data point consists of
    the 12 averaged channels.

    Args:
        data (np.ndarray): list of data points. Each data point is a time series
        with 12 channels of cepstrum coefficients.

    Returns:
        np.ndarray: the mean data point, consisting of the 12 averaged time series
    """

    # get the frequency of each time step (how often each time step was not padded)
    time_steps = get_time_steps(data)

    # to avoid division by 0
    time_steps = [1 if time_step == 0 else time_step for time_step in time_steps]

    # sum the values per channel
    summed_data = np.sum(data, axis=0)

    # calculate the mean per channel
    pattern_mean = (summed_data.transpose() / time_steps).transpose()
    return pattern_mean


def flatten_data(data: np.ndarray) -> np.ndarray:
    """
    Restructures the data points in the data array to vectors of length
    [channels x time steps] instead of matrices with [time steps] rows and
    [channels] columns.

    Args:
        data (np.ndarray): list of data points. Each data point is a time series
        with 12 channels of cepstrum coefficients.

    Returns:
        np.ndarray: same data array, but with flattened data points.
    """
    return np.array([data[idx].flatten() for idx in range(data.shape[0])])


def generate_class_matrix(n_data_points: int, n_classes: int) -> np.ndarray:
    """
    Generates the class matrix corresponding to the training data. The original
    training data consists 270 datapoints and 9 classes, where the first 30 data
    points are from class 1, the next 30 from class 2, etc. The class matrix is
    a list of one-hot encoded class-labels.

    Args:
        n_data_points (int): number of data points
        n_classes (int): number of classes

    Returns:
        np.ndarray: the class matrix, a list of one-hot encoded class-labels.
    """

    # should be 30 for the original training data
    points_per_class = n_data_points // n_classes

    return np.array(
        [
            [
                (
                    1
                    if data_point < (class_n + 1) * points_per_class
                    and data_point >= (class_n) * points_per_class
                    else 0
                )
                for class_n in range(n_classes)
            ]
            for data_point in range(n_data_points)
        ]
    )


def generate_test_class_matrix(n_data_points: int, n_classes: int) -> np.ndarray:
    """
    Generates the class matrix corresponding to the test data. The distribution
    of classes in the test data set is found on
    https://archive.ics.uci.edu/dataset/128/japanese+vowels.
    The class matrix is a list of one-hot encoded class-labels.

    Args:
        n_data_points (int): number of data points
        n_classes (int): number of classes

    Returns:
        np.ndarray: the class matrix, a list of one-hot encoded class-labels.
    """

    # data points per class (in order) in test data
    points_per_class = [31, 35, 88, 44, 29, 24, 40, 50, 29]

    return np.array(
        [
            [
                (
                    1
                    if data_point < sum(points_per_class[: idx + 1])
                    and data_point >= sum(points_per_class[:idx])
                    else 0
                )
                for idx in range(n_classes)
            ]
            for data_point in range(n_data_points)
        ]
    )


def remove_padding(signal: np.ndarray) -> np.ndarray:
    """
    Removes the padded values from a signal.

    Args:
        signal (np.ndarray): a (padded) time series with 12 channels of
        cepstrum coefficients.

    Returns:
        np.ndarray: the same signal as the input signal, but without the
        padded values (0's).
    """

    last_row = np.nonzero(signal)[0][-1] + 1  # last index of non-padded row
    signal = np.array([signal[:last_row, channel] for channel in range(N_CHANNELS)])
    return signal
