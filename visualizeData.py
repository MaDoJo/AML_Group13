from processing import get_pattern_mean

import numpy as np
import matplotlib.pyplot as plt
import math

def visualize_data_point(data_point: np.ndarray):
    if data_point.shape[1] != 12:
        print(f"expected data to have 12 columns, instead got {data_point.shape[1]}")
        return

    max_rows, n_dimensions = data_point.shape

    # last index of non-padded row
    last_row = np.nonzero(data_point)[0][-1] + 1

    # Create the x-axis (row indices)
    x_axis = np.arange(last_row)

    plt.figure(figsize=(12, 6))

    # Plot each of the 12 columns/dimensions as a separate line
    for channel in range(n_dimensions):
        plt.plot(x_axis, data_point[:last_row, channel], label=f'Channel {channel+1}')

    plt.xlim(0, max_rows)
    plt.title('Time Series Plot of a Recording', fontsize=14)
    plt.xlabel('Time Step', fontsize=12)
    plt.legend(loc='upper right', ncol=4, fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()


def make_plot(data_point, ax, data_class):
    n_dimensions = data_point.shape[1]

    # last index of non-padded row
    last_row = np.nonzero(data_point)[0][-1] + 1

    # Create the x-axis (row indices)
    x_axis = np.arange(last_row)

    # Plot each of the 12 columns/dimensions as a separate line
    for channel in range(n_dimensions):
        ax.plot(x_axis, data_point[:last_row, channel], label=f'Channel {channel+1}')

    ax.set_title(f'Mean pattern of class {data_class}', fontsize=8)
    ax.set_xlabel('Time Step', fontsize=6)
    ax.grid(True, linestyle='--', alpha=0.6)

def visualize_class_means(data):
    """
    Calculates and visualizes the mean pattern for each of the 9 classes
    in a 3x3 Matplotlib subplot figure.

    Args:
        data_array (np.ndarray): The input data array (270 data points).
    """
    # Total number of data points and classes
    N_TOTAL_POINTS = 270
    N_CLASSES = 9
    POINTS_PER_CLASS = N_TOTAL_POINTS // N_CLASSES  # Should be 30

    if data.shape[0] != N_TOTAL_POINTS:
        print(f"Warning: Expected {N_TOTAL_POINTS} data points, but got {data.shape[0]}.")

    # 1. Create the Matplotlib figure and 3x3 subplots
    fig, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(10, 7),
        sharex=True,
        sharey=True
    )
    # Flatten the 3x3 array of axes for easy iteration
    axes = axes.flatten()

    # 2. Iterate through each class
    for i in range(N_CLASSES):
        # Determine the start and end indices for the current class
        start_index = i * POINTS_PER_CLASS
        end_index = (i + 1) * POINTS_PER_CLASS

        # 3. Extract data for the current class
        class_data = data[start_index:end_index]

        # 4. Calculate the pattern mean
        mean_pattern = get_pattern_mean(class_data)

        # 5. Visualize the mean pattern on the corresponding subplot
        current_ax = axes[i]
        # Pass the subplot axis and the class index to the visualization function
        make_plot(mean_pattern, current_ax, i + 1)

    # 6. Add a super title and clean up layout
    fig.suptitle('Mean Data Patterns for the 9 Classes', fontsize=10, y=1.02)
    plt.tight_layout()
    plt.show()


def visualize_data_lengths(data_lengths):
    x_axis = np.arange(len(data_lengths))

    plt.figure(figsize=(12, 6))
    plt.plot(x_axis, data_lengths)
    plt.title('Number of Data Points at each Time Step', fontsize=14)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Number of data points', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()


def visualize_PC_variance(singular_values, log=False):
    x_axis = np.arange(len(singular_values))

    title = ("Log 10 " if log else "") + "Principal Component Variances"
    if log:
        singular_values = [math.log10(value) for value in singular_values]

    plt.figure(figsize=(12, 6))
    plt.plot(x_axis, singular_values)
    plt.title(title, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
