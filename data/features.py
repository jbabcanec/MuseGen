from music21 import converter
import numpy as np
import os

raw_folder = './raw'
processed_folder = './processed'

from validation.plot_intensity import get_intensity_profile

def extract_harmony_features(midi_path):
    score = converter.parse(midi_path)
    chords = score.chordify()
    chord_features = []

    for c in chords.recurse().getElementsByClass('Chord'):
        root = c.root().name
        quality = c.quality
        inversion = c.inversion()
        chord_info = f"{root}_{quality}_inv{inversion}"
        chord_features.append(chord_info)

    return chord_features

def append_features_to_npz(midi_file, npz_file):
    harmony_features = extract_harmony_features(midi_file)
    intensity_profile = get_intensity_profile(midi_file)

    with np.load(npz_file, allow_pickle=True) as data:
        existing_data = {key: data[key] for key in data.files}

    existing_data['features'] = {
        "harmony": harmony_features,
        "intensity_profile": intensity_profile
    }

    np.savez_compressed(npz_file, **existing_data)
    print(f"Features and intensity profile appended/updated in {npz_file}")

if __name__ == '__main__':
    # Set 'mode' to 'A' for processing all files or 'S' for a single file
    mode = 'S'  # Change this to 'A' if you want to process all files

    if mode == 'A':
        for midi_file in os.listdir(raw_folder):
            if midi_file.endswith('.mid') or midi_file.endswith('.midi'):
                midi_path = os.path.join(raw_folder, midi_file)
                npz_file_name = midi_file.replace('.mid', '').replace('.midi', '') + '_sequences.npz'
                npz_file_path = os.path.join(processed_folder, npz_file_name)
                
                if os.path.exists(npz_file_path):
                    append_features_to_npz(midi_path, npz_file_path)
                else:
                    print(f"No corresponding .npz file found for {midi_file} in {processed_folder}.")

    elif mode == 'S':
        # Specific MIDI file to process
        midi_file = 'beethoven_les_adieux_3.mid'
        midi_path = os.path.join(raw_folder, midi_file)
        npz_file_name = midi_file.replace('.mid', '').replace('.midi', '') + '_sequences.npz'
        npz_file_path = os.path.join(processed_folder, npz_file_name)
        
        if os.path.exists(npz_file_path):
            append_features_to_npz(midi_path, npz_file_path)
        else:
            print(f"No corresponding .npz file found for {midi_file} in {processed_folder}.")
