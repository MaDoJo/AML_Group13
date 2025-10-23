import os

import numpy as np

TRAIN_DATA_POINTS = 270
TEST_DATA_POINTS = 370
N_CHANNELS = 12
N_CLASSES = 9
MAX_LENGTH = 29


def load_data(
    file_path: str, num_data_points: int, padding_value: float = 0.0
) -> np.ndarray:
    """
    Loads the data from the data files, splits it into num_data_points individual
    data points, (which are time series with N_CHANNELS channels of cepstrum
    coefficients), pads them to the maximum sequence length (MAX_LENGTH), and
    returns a 3D NumPy array of shape (num_data_points, MAX_LENGTH, N_CHANNELS).

    Args:
        file_path (str): The path to the data file.
        num_data_points (int): The number of data points in the file.
        padding_value (float): The value used to pad shorter sequences.

    Returns:
        numpy.ndarray: The data from the data file as a NumPy array of shape
        (num_data_points, MAX_LENGTH, N_CHANNELS).
    """

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None

    # load all data from the file
    full_data = np.loadtxt(file_path)

    # separator rows consist of all ones
    is_separator = np.all(full_data == 1.0, axis=1)
    separator_indices = np.where(is_separator)[0]

    if len(separator_indices) != num_data_points:
        print(
            f"Warning: Found {len(separator_indices)} separator rows, expected \
              {num_data_points}."
        )

    # split the data into individual data points
    data_points = []
    start_idx = 0
    for sep_idx in separator_indices:
        data_segment = full_data[start_idx:sep_idx, :]
        data_points.append(data_segment)
        start_idx = sep_idx + 1

    if len(data_points) != num_data_points:
        print(
            f"Error: Number of extracted data points ({len(data_points)}) \
                  does not match expected ({num_data_points})."
        )
        return None

    # pad the signals to be all the same length (MAX_LENGTH)
    padded_data_points = []
    for arr in data_points:
        # create a padding array of the required size
        n_rows_to_pad = MAX_LENGTH - arr.shape[0]
        padding = np.full((n_rows_to_pad, N_CHANNELS), padding_value, dtype=arr.dtype)

        # add the padding to the the original array
        padded_arr = np.vstack([arr, padding])
        padded_data_points.append(padded_arr)

    # put the padded signals in a numpy array
    data_numpy_array = np.stack(padded_data_points)

    print(f"Successfully created a NumPy array with shape: {data_numpy_array.shape}")
    return data_numpy_array
