import numpy as np
import matplotlib.pyplot as plt

def visualize_data_point(data_point: np.ndarray):
    if data_point.shape[1] != 12:
        print(f"expected data to have 12 columns, instead got {data_point.shape[1]}")
        return

    max_rows, n_dimensions = data_point.shape

    # Create the x-axis (row indices)
    x_axis = np.arange(max_rows)

    plt.figure(figsize=(12, 6))

    # Plot each of the 12 columns/dimensions as a separate line
    for i in range(n_dimensions):
        plt.plot(x_axis, data_point[:, i], label=f'Channel {i+1}')

    plt.title('Time Series Plot of a Recording', fontsize=14)
    plt.xlabel('Time Step', fontsize=12)
    plt.legend(loc='upper right', ncol=4, fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('single_data_point_plot.png')
