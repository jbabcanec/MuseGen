import numpy as np
import os

processed_folder = './processed'
file_name = 'chpn_op25_e2_sequences.npz'  # Example file name

def check_npz(file_name):
    np.set_printoptions(suppress=True, threshold=np.inf, linewidth=200)  # Adjust print options for better readability

    file_path = os.path.join(processed_folder, file_name)
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return

    data = np.load(file_path, allow_pickle=True)

    print(f"Loaded data from {file_path}")

    # Loop through each key in the .npz file and print a sample from the first sequence
    for key in data.keys():
        print(f"\n{key.upper()} shape: {data[key].shape}")
        sequence = data[key][0]  # Take the first sequence
        print(f"Sample {key} data - First sequence (first 10 rows):")
        for line in sequence[:10]:  # Print the first 10 rows of the first sequence
            print(line)

if __name__ == '__main__':
    check_npz(file_name)
