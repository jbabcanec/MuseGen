import os
import mido
from pretty_midi import note_number_to_name

# Get the directory where this script is located
current_dir = os.path.dirname(__file__)

# Go up one directory from the current script's location
base_dir = os.path.join(current_dir, os.pardir)

# Define the raw and readable folders relative to the base directory
raw_directory = os.path.join(base_dir, 'raw')  # Path to the directory containing raw MIDI files
readable_directory = os.path.join(base_dir, 'readable')  # Directory to save readable output files

def read_midi(file_path, output_file):
    midi_data = mido.MidiFile(file_path)

    with open(output_file, 'w') as f:  # Use 'w' to write anew each time
        f.write(f"Processing {os.path.basename(file_path)}...\n\n")

        for i, track in enumerate(midi_data.tracks):
            f.write(f"Track {i}: {track.name}\n")
            absolute_time = 0  # Track the cumulative time to determine note start and end times
            
            for msg in track:
                absolute_time += msg.time  # Update the cumulative time
                if not msg.is_meta:
                    if msg.type == 'note_on' and msg.velocity > 0:
                        note_name = note_number_to_name(msg.note)
                        note_start = absolute_time
                        f.write(f"Note ON: Time {note_start}, Note Name: {note_name}, Velocity: {msg.velocity}\n")
                    elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                        note_name = note_number_to_name(msg.note)
                        note_end = absolute_time
                        f.write(f"Note OFF: Time {note_end}, Note Name: {note_name}\n")

def process_all_midi_files(raw_dir, output_dir):
    for file in os.listdir(raw_dir):
        if file.endswith('.midi') or file.endswith('.mid'):
            file_path = os.path.join(raw_dir, file)
        