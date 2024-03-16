import numpy as np
import os

# Get the directory where this script is located
current_dir = os.path.dirname(__file__)

# Go up one directory from the current script's location
base_dir = os.path.join(current_dir, os.pardir)

# Define the processed folder relative to the base directory
processed_folder = os.path.join(base_dir, 'processed')

def analyze_ranges(npz_file):
    with np.load(npz_file, allow_pickle=True) as data:
        # Initialize min/max values for this file
        min_pitch, max_pitch = float('inf'), float('-inf')
        min_velocity, max_velocity = float('inf'), float('-inf')
        min_time_delta, max_time_delta = float('inf'), float('-inf')

        for key in data.keys():
            # Check if the key corresponds to sequence data
            if 'sequences' in key or 'augmented' in key:  # Adjust based on your naming convention
                sequences = data[key]
                
                # Extract pitch, velocity, and time delta from all sequences under this key
                for seq in sequences:
                    pitches = seq[:, 1]
                    velocities = seq[:, 2]
                    time_deltas = seq[:, 3]

                    # Update min/max values for this key
                    min_pitch = min(min_pitch, np.min(pitches))
                    max_pitch = max(max_pitch, np.max(pitches))
                    min_velocity = min(min_velocity, np.min(velocities))
                    max_velocity = max(max_velocity, np.max(velocities))
                    min_time_delta = min(min_time_delta, np.min(time_deltas))
                    max_time_delta = max(max_time_delta, np.max(time_deltas))

    return min_pitch, max_pitch, min_velocity, max_velocity, min_time_delta, max_time_delta

# Initialize global min/max values
global_min_pitch, global_max_pitch = float('inf'), float('-inf')
global_min_velocity, global_max_velocity = float('inf'), float('-inf')
global_min_time_delta, global_max_time_delta = float('inf'), float('-inf')

# Process each .npz file in the processed folder
for npz_file in os.listdir(processed_folder):
    if npz_file.endswith('.npz'):
        full_path = os.path.join(processed_folder, npz_file)
        min_pitch, max_pitch, min_velocity, max_velocity, min_time_delta, max_time_delta = analyze_ranges(full_path)

        # Update global min/max values
        global_min_pitch = min(global_min_pitch, min_pitch)
        global_max_pitch = max(global_max_pitch, max_pitch)
        global_min_velocity = min(global_min_velocity, min_velocity)
        global_max_velocity = max(global_max_velocity, max_velocity)
        global_min_time_delta = min(global_min_time_delta, min_time_delta)
        global_max_time_delta = max(global_max_time_delta, max_time_delta)

# Print final report
print("Final Ranges Across All Sequences and Augmentations:")
print(f"Pitch Range: {global_min_pitch} to {global_max_pitch}")
print(f"Velocity Range: {global_min_velocity} to {global_max_velocity}")
print(f"Time Delta Range: {global_min_time_delta} to {global_max_time_delta}")
