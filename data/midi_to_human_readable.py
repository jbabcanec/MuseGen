import os
import shutil
import mido
from pretty_midi import PrettyMIDI, note_number_to_name

def read_midi(file_path, output_file):
    midi_data = mido.MidiFile(file_path)
    pretty_midi_data = PrettyMIDI(file_path)

    with open(output_file, 'a') as f:
        f.write(f"Processing {os.path.basename(file_path)}...\n")

        for i, track in enumerate(midi_data.tracks):
            f.write(f"Track {i}: {track.name}\n")
            for msg in track:
                if not msg.is_meta and msg.type == 'note_on':
                    note_name = note_number_to_name(msg.note)
                    note_start = msg.time
                    note_end = note_start
                    for msg_follow in track[msg.time:]:
                        if (msg_follow.type == 'note_off' or (msg_follow.type == 'note_on' and msg_follow.velocity == 0)) and msg_follow.note == msg.note:
                            note_end += msg_follow.time
                            break
                        note_end += msg_follow.time

                    velocity = msg.velocity
                    f.write(f"Note Start: {note_start}, Note End: {note_end}, Note Name: {note_name}, Velocity: {velocity}\n")

def process_all_midi_files(raw_dir, processed_dir, output_dir):
    for file in os.listdir(raw_dir):
        if file.endswith('.midi') or file.endswith('.mid'):
            file_path = os.path.join(raw_dir, file)
            output_file = os.path.join(output_dir, os.path.splitext(file)[0] + '.txt')
            read_midi(file_path, output_file)

if __name__ == "__main__":
    raw_directory = './raw'  # Path to the directory containing raw MIDI files
    processed_directory = './processed'  # Path to the directory where processed MIDI files should be moved
    output_directory = './processed'  # Directory to save output text files
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    process_all_midi_files(raw_directory, processed_directory, output_directory)
