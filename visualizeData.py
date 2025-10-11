from loadData import TRAIN_DATA_POINTS, N_CLASSES
from processing import get_pattern_mean

import numpy as np
import matplotlib.pyplot as plt
import math

def visualize_data_point(data_point: np.ndarray) -> None:
    """
    Generates a plot that displayes the timeseries of the 12 channels of a given
    data point. On the x-axis of the plot are the time steps and on the y-axis are
    the values of the cepstrum coefficients.

    Args:
        data_point (np.ndarray): The data point to plot. It is a time series with
        12 channels of cepstrum coefficients.
    """

    if data_point.shape[1] != 12:
        print(f"expected data to have 12 columns, instead got {data_point.shape[1]}")
        return

    max_rows, n_dimensions = data_point.shape

    # last index of non-padded row
    last_row = np.nonzero(data_point)[0][-1] + 1

    # the time steps are on the x-axis
    x_axis = np.arange(last_row)

    plt.figure(figsize=(12, 6))

    # plot each of the time series in the 12 channels as a separate line
    for channel in range(n_dimensions):
        plt.plot(x_axis, data_point[:last_row, channel], label=f'Channel {channel+1}')

    plt.xlim(0, max_rows)
    plt.title('Time Series Plot of a Recording', fontsize=14)
    plt.xlabel('Time Step', fontsize=12)
    plt.legend(loc='upper right', ncol=4, fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()


def make_subplot(pattern_mean: np.ndarray, ax, data_class: int) -> None:
    """
    Subroutine in the visualize_class_means function. Makes a subplot of
    a pattern mean on a given subplot ax. Similar to the visualize_data_point
    function.

    Args:
        pattern_mean (np.ndarray): the given pattern mean to plot. It is a time
        series with 12 channels of (averaged) cepstrum coefficients.
        ax (any): Matplotlib subplot axis to plot the pattern mean on.
        data_class (int): integer to indicate the class. Used for the subplot title.
    """

    n_dimensions = pattern_mean.shape[1]

    # last index of non-padded row
    last_row = np.nonzero(pattern_mean)[0][-1] + 1

    # the time steps are on the x-axis
    x_axis = np.arange(last_row)

    # plot each of the time series in the 12 channels as a separate line
    for channel in range(n_dimensions):
        ax.plot(x_axis, pattern_mean[:last_row, channel], label=f'Channel {channel+1}')

    # plot the pattern mean on the given subplot axis
    ax.set_title(f'Mean pattern of class {data_class}', fontsize=8)
    ax.set_xlabel('Time Step', fontsize=6)
    ax.grid(True, linestyle='--', alpha=0.6)


def visualize_class_means(data: np.ndarray) -> None:
    """
    Calculates and visualizes the mean pattern for each of the 9 classes
    in a 3x3 Matplotlib subplot figure.

    Args:
        data (np.ndarray): All data points from the original training data.
    """

    points_per_class = TRAIN_DATA_POINTS // N_CLASSES       # Should be 30

    if data.shape[0] != TRAIN_DATA_POINTS:
        print(f"Warning: Expected {TRAIN_DATA_POINTS} data points, but got \
              {data.shape[0]}.")

    # create the figure and 3x3 subplots, with the same x and y-axes
    fig, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(10, 7),
        sharex=True,
        sharey=True
    )

    # flatten the 3x3 array of axes for easy iteration
    axes = axes.flatten()

    # go over each class
    for i in range(N_CLASSES):
        # determine the start and end indices for the current class
        start_index = i * points_per_class
        end_index = (i + 1) * points_per_class

        # the data for the current class
        class_data = data[start_index:end_index]

        # get the pattern mean
        mean_pattern = get_pattern_mean(class_data)

        # select the corresponding subplot
        current_ax = axes[i]

        # plot the pattern mean
        make_subplot(mean_pattern, current_ax, i + 1)

    fig.suptitle('Mean Data Patterns for the 9 Classes', fontsize=10, y=1.02)
    plt.tight_layout()
    plt.show()


def visualize_data_lengths(data_lengths: np.ndarray) -> None:
    """
    Plots the distribution of the length of the time series. On the x-axis
    of the resulting plot are the number of (non-zero) time steps. On the
    y-axis are the number of data points (time series) that have at least
    this number of time steps.

    Args:
        data_lengths (np.ndarray): a list containing the frequencies of the
        use of all time steps of a certain set of data points (e.g. the 
        training data)
    """

    x_axis = np.arange(len(data_lengths))

    plt.figure(figsize=(12, 6))
    plt.plot(x_axis, data_lengths)
    plt.title('Number of Data Points at each Time Step', fontsize=14)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Number of data points', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()


def visualize_PC_variance(singular_values: np.ndarray, log: bool=False) -> None:
    """
    Plots the Principal Component (PC) variances in descending order. On the
    x-axis are the PCs in descending order from higest variance to lowest 
    variance. On the y-axis is the (log) variance of each PC.

    Args:
        singular_values (np.ndarray): A list with the variances of the PCs in
        descending order
        log (bool): determines if the raw PC variances will be plotted 
        (log == False) or the log of the PC variances (log == True).
    """

    x_axis = np.arange(len(singular_values))

    # generate the correct title
    title = ("Log 10 " if log else "") + "Principal Component Variances"

    # get the log of the PC variances if log == True
    if log:
        singular_values = [math.log10(value) for value in singular_values]

    plt.figure(figsize=(12, 6))
    plt.plot(x_axis, singular_values)
    plt.title(title, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
