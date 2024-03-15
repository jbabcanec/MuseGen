from music21 import converter
import numpy as np
import os

raw_folder = './raw'
processed_folder = './processed'

def extract_harmony_features(midi_path):
    score = converter.parse(midi_path)
    chords = score.chordify()
    chord_progressions = [c.root().name for c in chords.recurse().getElementsByClass('Chord')]
    return chord_progressions

def extract_rhythm_features(midi_path):
    score = converter.parse(midi_path)
    durations = [str(n.duration.quarterLength) for n in score.recurse().getElementsByClass(['Note', 'Rest'])]
    return durations

def extract_melody_features(midi_path):
    score = converter.parse(midi_path)
    melody = score.parts[0].recurse().getElementsByClass('Note')
    melody_sequence = [str(n.pitch.midi) for n in melody]
    return melody_sequence

def append_features_to_npz(midi_file, npz_file):
    # Extract features from the MIDI file in the 'raw' folder
    harmony_features = extract_harmony_features(midi_file)
    rhythm_features = extract_rhythm_features(midi_file)
    melody_features = extract_melody_features(midi_file)

    # Load existing data from the .npz file in the 'processed' folder, which includes augmented data
    with np.load(npz_file, allow_pickle=True) as data:
        # Retain all existing data
        existing_data = {key: data[key] for key in data.files}

    # Append the extracted features to the existing data
    existing_data['features'] = {
        "harmony": harmony_features,
        "rhythm": rhythm_features,
        "melody": melody_features
    }

    # Save all the data back to the .npz file, preserving augmented data
    np.savez_compressed(npz_file, **existing_data)
    print(f"Features appended to {npz_file}")

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