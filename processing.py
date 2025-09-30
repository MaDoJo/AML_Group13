import numpy as np

def get_time_steps(data):
    time_steps = np.zeros(29)

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

def normalize(data, pattern_mean):
    n_time_steps = data.shape[1]

    for data_point in data:
        for time_step in range(n_time_steps):
            if np.count_nonzero(data_point[time_step]) > 0:
                data_point[time_step] -= pattern_mean[time_step]

def flatten_data(data):
    return np.array([data[idx].flatten() for idx in range(data.shape[0])])

def SVD(data):
    C = (1 / data.shape[0]) * np.matmul(data.transpose(), data)

    # only return U and S (and not U', since it is redundant)
    return np.linalg.svd(C)[:2]

def determine_cutoff(variance_vector, wanted_variance):
    cutoff = 1
    while (sum(variance_vector[:cutoff]) / sum(variance_vector)) * 100 < wanted_variance:
        cutoff += 1

    return cutoff

def generate_class_matrix(n_data_points, n_classes):
    return np.array([[1 if data_point < (class_n + 1) * 30 and data_point >= (class_n) * 30 else 0 for class_n in range(n_classes)] for data_point in range(n_data_points)])

def compute_regression_classifier(feature_vectors, class_matrix):
    return np.matmul(np.linalg.inv(np.matmul(feature_vectors.transpose(), feature_vectors)), np.matmul(feature_vectors.transpose(), class_matrix))

def compute_MSE(regression_classifier, feature_vectors, class_matrix):
    return sum([np.linalg.norm(class_matrix[idx] - np.matmul(regression_classifier.transpose(), feature_vector))**2 for idx, feature_vector in enumerate(feature_vectors)]) / len(feature_vectors)

def compute_mismatch(regression_classifier, feature_vectors, class_matrix):
    return sum([0 if np.argmax(class_matrix[idx]) == np.argmax(np.matmul(regression_classifier.transpose(), feature_vector)) else 1 for idx, feature_vector in enumerate(feature_vectors)]) / len(feature_vectors)

def generate_test_class_matrix(n_data_points, n_classes):
    points_per_class = [31, 35, 88, 44, 29, 24, 40, 50, 29]
    return np.array([[1 if data_point < sum(points_per_class[:idx + 1]) and data_point >= sum(points_per_class[:idx]) else 0 for idx in range(n_classes)] for data_point in range(n_data_points)])
