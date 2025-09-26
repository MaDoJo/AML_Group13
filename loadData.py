import numpy as np
import os

def load_data(file_path: str, num_data_points: int, padding_value: float=0.0):
    """
    Loads the data from the ae.train file, splits it into num_data_points individual data points 
    (sequences), pads them to the maximum sequence length (max_rows), and returns 
    a 3D NumPy array of shape (num_data_points, max_rows, 12).

    Args:
        file_path (str): The path to the ae.train file.
        num_data_points (int): The expected number of data points.
        padding_value (float): The value used to pad shorter sequences (typically 0.0 or np.nan).

    Returns:
        numpy.ndarray: A NumPy array of shape (num_data_points, max_rows, 12).
    """
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None

    try:
        # 1. Load all data from the file
        full_data = np.loadtxt(file_path)
        N_DIMENSIONS = full_data.shape[1] # Should be 12

        # 2. Identify the separator rows (the row of 12 ones)
        # Check if all elements in a row are exactly 1.0
        is_separator = np.all(full_data == 1.0, axis=1)
        separator_indices = np.where(is_separator)[0]

        if len(separator_indices) != num_data_points:
            print(f"Warning: Found {len(separator_indices)} separator rows, expected {num_data_points}.")

        # 3. Split the data into individual sequences
        data_points = []
        start_idx = 0
        for sep_idx in separator_indices:
            # Extract the segment from the current start index up to the separator index
            data_segment = full_data[start_idx:sep_idx, :]
            data_points.append(data_segment)
            # The next segment starts after the separator row
            start_idx = sep_idx + 1
        
        if len(data_points) != num_data_points:
             print(f"Error: Number of extracted data points ({len(data_points)}) does not match expected ({num_data_points}).")
             return None

        # 4. Calculate max_rows (maximum sequence length)
        max_rows = max(arr.shape[0] for arr in data_points)
        print(f"Maximum number of rows of all datapoints: {max_rows}")

        # 5. Pad and Collect
        padded_data_points = []
        for arr in data_points:
            n_rows_to_pad = max_rows - arr.shape[0]
            
            # Create a padding array of the required size
            padding = np.full((n_rows_to_pad, N_DIMENSIONS), padding_value, dtype=arr.dtype)
            
            # Concatenate the original array and the padding at the bottom
            padded_arr = np.vstack([arr, padding])
            padded_data_points.append(padded_arr)

        # 6. Stack the resulting num_data_points padded arrays
        data_numpy_array = np.stack(padded_data_points)

        print(f"Successfully created a NumPy array with shape: {data_numpy_array.shape}")
        return data_numpy_array

    except Exception as e:
        print(f"An error occurred while processing the file: {e}")
        return None


if __name__ == "__main__":
    # Execute the function to create the 3D NumPy array
    data_numpy_array = load_data()

    # You can now work with the 'data_numpy_array'
    if data_numpy_array is not None:
        print("\nShape of the final array:", data_numpy_array.shape)
        
        # Example: Print the shape of the first data point (sequence 0)
        print("Shape of the first data point (should be (max_rows, 12)):", data_numpy_array[0].shape)

        # Example: Print the first few rows of the first sequence
        print("\nFirst 5 rows of the first sequence:")
        print(data_numpy_array[0])
