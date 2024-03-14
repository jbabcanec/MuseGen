import numpy as np
import os

processed_folder = './processed'

def analyze_ranges(npz_file):
    with np.load(npz_file, allow_pickle=True) as data:
        next_events = data['next_events']  # Assuming this key exists based on your preprocessing script

    # Extract pitch, velocity, and time delta from next_events
    pitches = next_events[:, 1]
    velocities = next_events[:, 2]
    time_deltas = next_events[:, 3]

    return np.min(pitches), np.max(pitches), np.min(velocities), np.max(velocities), np.min(time_deltas), np.max(time_deltas)

# Initialize min/max values
min_pitch = float('inf')
max_pitch = float('-inf')
min_velocity = float('inf')
max_velocity = float('-inf')
min_time_delta = float('inf')
max_time_delta = float('-inf')

# Process each .npz file in the processed folder
for npz_file in os.listdir(processed_folder):
    if npz_file.endswith('.npz'):
        full_path = os.path.join(processed_folder, npz_file)
        pitch_min, pitch_max, velocity_min, velocity_max, time_delta_min, time_delta_max = analyze_ranges(full_path)

        # Update overall min/max values
        min_pitch = min(min_pitch, pitch_min)
        max_pitch = max(max_pitch, pitch_max)
        min_velocity = min(min_velocity, velocity_min)
        max_velocity = max(max_velocity, velocity_max)
        min_time_delta = min(min_time_delta, time_delta_min)
        max_time_delta = max(max_time_delta, time_delta_max)

# Print final report
print("Final Ranges:")
print(f"Pitch Range: {min_pitch} to {max_pitch}")
print(f"Velocity Range: {min_velocity} to {max_velocity}")
print(f"Time Delta Range: {min_time_delta} to {max_time_delta}")
