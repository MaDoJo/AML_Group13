import numpy as np
from scipy.interpolate import interp1d

from src.utils.config import N_CLASSES
from src.utils.processing import (add_padding, generate_class_matrix,
                                  remove_padding)


def augment_data(
    data,
    augmentations_per_sample=3,
    noise_std=0.05
) -> np.ndarray:
    """
    Perform 3 augmentation techniques on cepstral time series data arranged by class.
    Each original sample produces 3 new augmented samples.

    Args:
        data : np.ndarray, shape (N, T, F)
            Original training data (e.g. (270, 29, 12))
        augmentations_per_sample : int
            Number of augmentations to apply per sample (default 3)
        noise_std : float
            Standard deviation factor for Gaussian noise
        scale_range : tuple
            Range (min, max) for random scaling
        time_warp_range : tuple
            Range (min, max) for random time-warping factor

    Returns:
        augmented_train_data : pd.DataFrame
            DataFrame with 'data' and 'class_id' columns
    """

    points_per_class = data.shape[0] // N_CLASSES
    augmented_samples = []

    for class_id in range(N_CLASSES):
        start = class_id * points_per_class
        end = (class_id + 1) * points_per_class
        class_data = data[start:end]

        for x_original in class_data:
            x = remove_padding(x_original).transpose()

            for _ in range(augmentations_per_sample):
                # Additive Gaussian noise
                noise = np.random.normal(0, noise_std * np.std(x), x.shape)
                x_noise = add_padding(x + noise)
                augmented_samples.extend([x_noise])

            augmented_samples.extend([x_original])

    augmented_samples = np.array(augmented_samples)
    all_class_ids = generate_class_matrix(augmented_samples.shape[0], N_CLASSES)

    return augmented_samples, all_class_ids
