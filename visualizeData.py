import numpy as np
import matplotlib.pyplot as plt

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
    plt.savefig('single_data_point_plot.png')
