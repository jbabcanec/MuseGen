import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor

try:
    from validation.plot_intensity import get_intensity_profile
except Exception:
    # Importing within try/except so this script can run even if optional
    # research utilities are not available. The augmentation functions will
    # still work without intensity based scaling.
    get_intensity_profile = None

raw_folder = './raw'
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

def velocity_scale_sequence(sequence, scale):
    """Scale the velocity of each event by ``scale`` while keeping values in
    the valid MIDI range.
    """
    scaled_sequence = []
    for event in sequence:
        event_type, pitch, velocity, delta_t = event
        new_velocity = int(np.clip(velocity * scale, 0, 127))
        scaled_sequence.append([int(event_type), int(pitch), new_velocity, int(delta_t)])
    return np.array(scaled_sequence, dtype=int)

def apply_intensity_profile(midi_path, sequences):
    """Apply the intensity profile of ``midi_path`` to ``sequences`` by scaling
    velocities. The profile is interpolated to the length of each sequence."""
    if get_intensity_profile is None:
        return None
    profile = get_intensity_profile(midi_path)
    if not profile:
        return None
    augmented = []
    for seq in sequences:
        scaled_seq = []
        for i, event in enumerate(seq):
            idx = int(i / len(seq) * (len(profile) - 1))
            scale = 0.5 + 0.5 * profile[idx]
            event_type, pitch, velocity, delta_t = event
            new_velocity = int(np.clip(velocity * scale, 0, 127))
            scaled_seq.append([int(event_type), int(pitch), new_velocity, int(delta_t)])
        augmented.append(np.array(scaled_seq, dtype=int))
    return np.array(augmented)


def augment_data(file_path, semitone_shifts, time_factors, velocity_factors=None, apply_intensity=True):
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

    # Velocity scaling
    if velocity_factors:
        for v_scale in velocity_factors:
            scaled = [velocity_scale_sequence(seq, v_scale) for seq in sequences]
            label = f"velocity_scaled_{v_scale}"
            augmented_data[label] = np.array(scaled)

    # Intensity profile based scaling if possible
    if apply_intensity:
        midi_name = os.path.basename(file_path).replace('_sequences.npz', '.mid')
        midi_path = os.path.join(raw_folder, midi_name)
        if os.path.exists(midi_path):
            intensity_scaled = apply_intensity_profile(midi_path, sequences)
            if intensity_scaled is not None:
                augmented_data['intensity_scaled'] = intensity_scaled

    # Append augmented data with labels, including the original sequences, to the .npz file
    np.savez_compressed(file_path, next_events=next_events, **augmented_data)
    print(f"Augmented data with labels appended to {file_path}")


def process_file(file_name):
    if file_name.endswith('.npz'):
        file_path = os.path.join(processed_folder, file_name)
        augment_data(
            file_path,
            semitone_shifts=[2, -2],
            time_factors=[1.5, 0.75],
            velocity_factors=[1.2, 0.8],
            apply_intensity=True,
        )

if __name__ == '__main__':
    midi_files = [f for f in os.listdir(processed_folder) if f.endswith('.npz')]

    # Use ThreadPoolExecutor to parallelize the augmentation process
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        executor.map(process_file, midi_files)
