import numpy as np
import os

processed_folder = './processed'
file_name = 'chpn_op25_e2_sequences.npz'  # Placeholder file name

def check_npz(file_name):
    np.set_printoptions(suppress=True)  # Suppress scientific notation

    file_path = os.path.join(processed_folder, file_name)
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return

    data = np.load(file_path)
    sequences = data['sequences']
    next_events = data['next_events']

    print(f"Loaded data from {file_path}")
    print(f"Sequences shape: {sequences.shape}")
    print(f"Next events shape: {next_events.shape}")

    # Optionally, print a small sample of the data
    print("Sample sequences data:", sequences[:1])  # Print first sequence
    print("Sample next events data:", next_events[:1])  # Print first next event

if __name__ == '__main__':
    check_npz(file_name)
