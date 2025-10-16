import numpy as np
import os

TRAIN_DATA_POINTS = 270
TEST_DATA_POINTS = 370
N_CLASSES = 9
MAX_LENGTH = 29

def load_data(file_path: str, num_data_points: int, padding_value: float=0.0):
    """
    Loads the data from the file, splits it into num_data_points individual data points 
    using blank lines as separators, pads them to the maximum sequence length
    and returns a 3D NumPy array of shape (num_data_points, MAX_LENGTH, 12).
    """
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None

    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        data_points = []
        current_sequence = []
        
        for line in lines:
            line = line.strip()
            
            if not line:
                if current_sequence:
                    data_points.append(np.array(current_sequence))
                    current_sequence = []
            else:
                # parse line into 12 float values
                values = [float(x) for x in line.split()]
                if len(values) == 12:
                    current_sequence.append(values)
        
        if current_sequence:
            data_points.append(np.array(current_sequence))
        
        if len(data_points) != num_data_points:
            print(f"Warning: Found {len(data_points)} sequences, expected {num_data_points}.")

        # print(f"max number of rows of all datapoints: {MAX_LENGTH}")

        # padding
        padded_data_points = []
        for arr in data_points:
            n_rows_to_pad = MAX_LENGTH - arr.shape[0]
            
            if n_rows_to_pad > 0:
                padding = np.full((n_rows_to_pad, 12), padding_value, dtype=arr.dtype)
                padded_arr = np.vstack([arr, padding])
            else:
                padded_arr = arr[:MAX_LENGTH]
            
            padded_data_points.append(padded_arr)

        data_numpy_array = np.stack(padded_data_points)
        return data_numpy_array

    except Exception as e:
        print(f"An error occurred while processing the file: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    data_numpy_array = load_data()

    if data_numpy_array is not None:
        print("\nShape of the final array:", data_numpy_array.shape)
        
        print("Shape of the first data point (should be (29, 12)):", data_numpy_array[0].shape)

        print("\nFirst 5 rows of the first sequence:")
        print(data_numpy_array[0])