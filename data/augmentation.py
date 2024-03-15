import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor

processed_folder = './processed'

def transpose_sequence(sequence, semitone_shift):
    transposed_sequence = []
    for event in sequence:
        event_type, pitch, velocity, delta_t = event
        if event_type in [1, 2]:  # Note on or note on with velocity below threshold
            transposed_pitch = np.clip(pitch + semitone_shift, 0, 127)
        else:
            transposed_pitch = pitch
        # Ensure the result is an integer
        transposed_sequence.append([int(event_type), int(transposed_pitch), int(velocity), int(delta_t)])
    return np.array(transposed_sequence, dtype=int)

def augment_sequence(sequence, factor):
    augmented_sequence = []
    for event in sequence:
        event_type, pitch, velocity, delta_t = event
        augmented_delta_t = delta_t * factor
        # Ensure the result is an integer
        augmented_sequence.append([int(event_type), int(pitch), int(velocity), int(augmented_delta_t)])
    return np.array(augmented_sequence, dtype=int)

def diminish_sequence(sequence, factor):
    diminished_sequence = []
    for event in sequence:
        event_type, pitch, velocity, delta_t = event
        diminished_delta_t = delta_t / factor if factor != 0 else delta_t
        # Ensure the result is an integer, use round() before int() to handle division properly
        diminished_sequence.append([int(event_type), int(pitch), int(velocity), int(round(diminished_delta_t))])
    return np.array(diminished_sequence, dtype=int)


def augment_data(file_path, semitone_shifts, time_factors):
    with np.load(file_path, allow_pickle=True) as data:
        sequences = data['sequences']
        next_events = data['next_events']

    augmented_data = {}

    # Label and store the original sequences for clarity
    augmented_data['original'] = sequences

    # Transposition
    for shift in semitone_shifts:
        augmented_sequences = [transpose_sequence(seq, shift) for seq in sequences]
        label = f"transposed_{shift}"
        augmented_data[label] = np.array(augmented_sequences)

    # Time augmentation and diminution
    for factor in time_factors:
        augmented_sequences = [augment_sequence(seq, factor) for seq in sequences if factor > 1] + \
                              [diminish_sequence(seq, factor) for seq in sequences if factor < 1 and factor > 0]
        label = "augmented" if factor > 1 else "diminished"
        augmented_data[f"{label}_{factor}"] = np.array(augmented_sequences)

    # Append augmented data with labels, including the original sequences, to the .npz file
    np.savez_compressed(file_path, next_events=next_events, **augmented_data)
    print(f"Augmented data with labels appended to {file_path}")


def process_file(file_name):
    if file_name.endswith('.npz'):
        file_path = os.path.join(processed_folder, file_name)
        augment_data(file_path, semitone_shifts=[2, -2], time_factors=[1.5, 0.75])

if __name__ == '__main__':
    midi_files = [f for f in os.listdir(processed_folder) if f.endswith('.npz')]

    # Use ThreadPoolExecutor to parallelize the augmentation process
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        executor.map(process_file, midi_files)