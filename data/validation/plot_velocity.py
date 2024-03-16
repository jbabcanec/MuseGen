import matplotlib.pyplot as plt
import numpy as np
from music21 import midi
from scipy.ndimage import gaussian_filter1d
import os

# Get the directory where this script is located
current_dir = os.path.dirname(__file__)

# Go up one directory from the current script's location
base_dir = os.path.join(current_dir, os.pardir)

# Define the raw folder relative to the base directory
raw_folder = os.path.join(base_dir, 'raw')  # Adjusted to go up a directory

def normalize_and_average_velocities(velocities_list, target_length=100):
    normalized_velocities = []
    for velocities in velocities_list:
        # Normalize the length of the velocity list to the target length
        indices = np.linspace(0, len(velocities)-1, num=target_length, dtype=int)
        normalized_velocities.append([velocities[i] for i in indices])
    
    # Calculate the average velocities
    avg_velocities = np.mean(normalized_velocities, axis=0)
    return avg_velocities

def plot_velocity(midi_file_path=None):
    velocities_list = []

    if midi_file_path:
        # If a specific MIDI file is specified, only process that file
        midi_files = [midi_file_path]
    else:
        # If no specific file is specified, process all MIDI files in the directory
        midi_files = [os.path.join(raw_folder, f) for f in os.listdir(raw_folder) if f.endswith(('.mid', '.midi'))]

    for file_path in midi_files:
        mf = midi.MidiFile()
        mf.open(file_path)
        mf.read()
        mf.close()

        # Extract notes and their velocities for each MIDI file
        velocities = []
        for track in mf.tracks:
            for event in track.events:
                if isinstance(event, midi.MidiEvent) and event.isNoteOn() and event.velocity > 0:
                    velocities.append(event.velocity)
        velocities_list.append(velocities)

    # Normalize and average velocities if more than one MIDI file is processed
    if len(velocities_list) > 1:
        velocities = normalize_and_average_velocities(velocities_list)
    else:
        velocities = velocities_list[0]

    # Smooth the velocities
    smooth_vel_1 = gaussian_filter1d(velocities, sigma=2)  # First level of smoothing
    smooth_vel_2 = gaussian_filter1d(velocities, sigma=5)  # Second level of smoothing

    # Plotting
    plt.figure(figsize=(15, 6))

    plt.plot(np.linspace(0, 100, len(velocities)), velocities, label='Raw Velocities', color='blue')
    plt.plot(np.linspace(0, 100, len(smooth_vel_1)), smooth_vel_1, label='Smoothed Velocities (Sigma=2)', color='red')
    plt.plot(np.linspace(0, 100, len(smooth_vel_2)), smooth_vel_2, label='Smoothed Velocities (Sigma=5)', color='green')

    plt.title('MIDI Velocities')
    plt.ylabel('Velocity')
    plt.xlabel('Normalized Time (0-100)')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    file_name = None  # Set to None to process all MIDI files
    midi_path = os.path.join(raw_folder, file_name) if file_name else None
    plot_velocity(midi_path)
