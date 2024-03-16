from music21 import converter
import numpy as np
import os

raw_folder = './raw'
processed_folder = './processed'

def extract_harmony_features(midi_path):
    score = converter.parse(midi_path)
    chords = score.chordify()
    chord_features = []

    for c in chords.recurse().getElementsByClass('Chord'):
        # Extract the root note, chord quality, and inversion
        root = c.root().name
        quality = c.quality
        inversion = c.inversion()

        # Format the chord information. Example: "C_major_inv1" for C major chord in first inversion
        chord_info = f"{root}_{quality}_inv{inversion}"
        chord_features.append(chord_info)

    return chord_features

def append_features_to_npz(midi_file, npz_file):
    harmony_features = extract_harmony_features(midi_file)

    with np.load(npz_file, allow_pickle=True) as data:
        existing_data = {key: data[key] for key in data.files}

    # Check if 'features' already exists and log a message
    if 'features' in existing_data:
        print(f"'features' section already exists in {npz_file} and will be overwritten.")

    existing_data['features'] = {
        "harmony": harmony_features,
    }

    np.savez_compressed(npz_file, **existing_data)
    print(f"Features appended/updated in {npz_file}")


if __name__ == '__main__':
    for midi_file in os.listdir(raw_folder):
        if midi_file.endswith('.mid') or midi_file.endswith('.midi'):
            midi_path = os.path.join(raw_folder, midi_file)
            # Construct the corresponding .npz file path in the 'processed' folder
            npz_file_name = midi_file.replace('.mid', '').replace('.midi', '') + '_sequences.npz'
            npz_file_path = os.path.join(processed_folder, npz_file_name)
            
            if os.path.exists(npz_file_path):
                append_features_to_npz(midi_path, npz_file_path)
            else:
                print(f"No corresponding .npz file found for {midi_file} in {processed_folder}.")