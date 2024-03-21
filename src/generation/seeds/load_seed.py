import numpy as np
import random
from pathlib import Path

# Function to load a random seed sequence from a .npz file in the sample_dir
def load_random_seed(sample_dir, sequence_length=100):
    npz_files = list(Path(sample_dir).glob('*.npz'))

    if not npz_files:
        raise ValueError("No .npz files found in the specified directory.")

    random_npz = random.choice(npz_files)
    print(f"Loading seed sequence from: {random_npz}")

    with np.load(random_npz) as data:
        # Assuming 'sequences' is a 3D array ([num_sequences, sequence_length, num_features])
        # and you want to select one sequence (2D) from it
        all_sequences = data['sequences']

        # Select a single sequence randomly or use a specific selection criteria
        selected_sequence_index = random.randint(0, all_sequences.shape[0] - 1)
        seed_sequence = all_sequences[selected_sequence_index]

        # Adjust the sequence to have the desired length
        if seed_sequence.shape[0] > sequence_length:
            # Truncate the sequence if it's longer than the desired length
            seed_sequence = seed_sequence[:sequence_length, :]
        elif seed_sequence.shape[0] < sequence_length:
            # Pad the sequence with zeros if it's shorter than the desired length
            padding = np.zeros((sequence_length - seed_sequence.shape[0], seed_sequence.shape[1]))
            seed_sequence = np.vstack([seed_sequence, padding])

    return seed_sequence
