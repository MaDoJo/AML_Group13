import numpy as np
from scipy.interpolate import interp1d

from src.utils.config import N_CLASSES
from src.utils.processing import remove_padding, add_padding, generate_class_matrix


def augment_data(
    data,
    augmentations_per_sample=3,
    noise_std=0.01,
    scale_range=(0.9, 1.1),
    time_warp_range=(0.9, 1.1),
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
    N, T, F = data.shape
    points_per_class = N // N_CLASSES

    augmented_samples = []
    augmented_class_ids = []

    for class_id in range(N_CLASSES):
        start = class_id * points_per_class
        end = (class_id + 1) * points_per_class
        class_data = data[start:end]

        for x in class_data:
            x = remove_padding(x).transpose()
            # 1. Additive Gaussian noise
            noise = np.random.normal(0, noise_std * np.std(x), x.shape)
            x_noise = add_padding(x + noise)

            # 2. Random scaling
            scale = np.random.uniform(*scale_range)
            x_scaled = add_padding(x * scale)

            # 3. Time warping
            factor = np.random.uniform(*time_warp_range)
            t_original = np.arange(x.shape[0])
            f = interp1d(t_original, x, axis=0, fill_value="extrapolate")
            x_warped = add_padding(f(np.linspace(0, x.shape[0] - 1, x.shape[0])))

            augmented_samples.extend([x_noise, x_scaled, x_warped])
            augmented_class_ids.extend([class_id] * 3)

    # combine with originals
    all_data = np.concatenate([data, np.stack(augmented_samples)], axis=0)
    all_class_ids = generate_class_matrix(all_data.shape[0], N_CLASSES)

    return all_data, all_class_ids
