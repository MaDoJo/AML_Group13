import numpy as np

def get_time_steps(data):
    time_steps = np.zeros(data.shape[1])

    for data_point in data:
        # list of length max-time-steps lenght (26 for training data) with 1's for
        # non-padded time steps and 0's for all-0-valued (padded) time steps
        mask = [1 if np.count_nonzero(time_step) > 0 else 0 for time_step in data_point]
        time_steps += mask

    return time_steps

def get_pattern_mean(data):
    summed_data = np.sum(data, axis=0)
    time_steps = get_time_steps(data)

    # to avoid division by 0
    time_steps = [1 if time_step == 0 else time_step for time_step in time_steps]
    pattern_mean = (summed_data.transpose() / time_steps).transpose()
    return pattern_mean

def normalize(data):
    n_time_steps = data.shape[1]
    pattern_mean = get_pattern_mean(data)

    for data_point in data:
        for time_step in range(n_time_steps):
            if np.count_nonzero(data_point[time_step]) > 0:
                data_point[time_step] -= pattern_mean[time_step]
