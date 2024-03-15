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
        transposed_sequence.append([event_type, transposed_pitch, velocity, delta_t])
    return transposed_sequence

def augment_sequence(sequence, factor):
    augmented_sequence = []
    for event in sequence:
        event_type, pitch, velocity, delta_t = event
        augmented_delta_t = delta_t * factor
        augmented_sequence.append([event_type, pitch, velocity, augmented_delta_t])
    return augmented_sequence

def diminish_sequence(sequence, factor):
    diminished_sequence = []
    for event in sequence:
        event_type, pitch, velocity, delta_t = event
        diminished_delta_t = delta_t / factor if factor != 0 else delta_t
        diminished_sequence.append([event_type, pitch, velocity, diminished_delta_t])
    return diminished_sequence


def augment_data(file_path, semitone_shifts, time_factors):
    with np.load(file_path, allow_pickle=True) as data:
        sequences = data['sequences']
        next_events = data['next_events']

    augmented_data = {}

    # Transposition
    for shift in semitone_shifts:
        augmented_sequences = []
        for seq in sequences:
            transposed_seq = transpose_sequence(seq, shift)
            augmented_sequences.append(transposed_seq)
        label = f"transposed_{shift}"  # Label indicating the transposition
        augmented_data[label] = np.array(augmented_sequences)

    # Time augmentation and diminution
    for factor in time_factors:
        augmented_sequences = []
        for seq in sequences:
            if factor > 1:  # Augmentation
                augmented_seq = augment_sequence(seq, factor)
                augmented_sequences.append(augmented_seq)
            elif factor < 1 and factor > 0:  # Diminution
                diminished_seq = diminish_sequence(seq, factor)
                augmented_sequences.append(diminished_seq)
        label = "augmented" if factor > 1 else "diminished"  # Label indicating time change
        augmented_data[f"{label}_{factor}"] = np.array(augmented_sequences)

    # Append augmented data with labels to the .npz file
    np.savez_compressed(file_path, sequences=sequences, next_events=next_events, **augmented_data)
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