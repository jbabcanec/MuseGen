import os
import random
import mido
import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures
from scipy.interpolate import UnivariateSpline

#------------------------------------------------------------
# Preprocessing 
#------------------------------------------------------------

def select_midi_files(midi_files, midi_num):
    if midi_num == 'all':
        return midi_files
    else:
        choice = random.sample(midi_files, min(midi_num, len(midi_files)))
        print(choice)
        return choice

def combine_tracks(midi_file):
    mid = mido.MidiFile(midi_file)
    combined = mido.merge_tracks(mid.tracks)  # Combining all tracks into one
    new_mid = mido.MidiFile()  # Create a new MidiFile
    new_mid.tracks.append(combined)  # Append the combined track
    return new_mid

#------------------------------------------------------------
# Profiles
#------------------------------------------------------------

def velocity_profile(mid_file):
    velocities = []  # List to store the velocity of each note-on event
    for msg in mid_file.tracks[0]:
        if msg.type == 'note_on' and msg.velocity > 0:
            # Scale the velocity from 0 to 1 and append to the list
            scaled_velocity = msg.velocity #/ 127
            velocities.append(scaled_velocity)
    return velocities

def pitch_density_profile(mid_file):
    density_profile = []
    current_notes = set()  # Set to track currently active notes

    for msg in mid_file.tracks[0]:
        if msg.type == 'note_on' and msg.velocity > 0:
            current_notes.add(msg.note)  # Add note to the set of active notes
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            current_notes.discard(msg.note)  # Remove note from the set of active notes
        
        # Append the current number of active notes to the density profile
        density_profile.append(len(current_notes))

    return density_profile

def interval_tension(interval):
    # Define intervals that typically create tension
    tension_intervals = [1, 6, 10, 11]  # Minor second, tritone, major seventh
    return interval in tension_intervals

def calculate_tension(active_notes):
    tension_score = 0
    for note1 in active_notes:
        for note2 in active_notes:
            if note1 != note2:
                interval = abs(note1 - note2) % 12  # Calculate the interval in semitones, considering octave equivalence
                if interval_tension(interval):
                    tension_score += 1
    return tension_score

def tension_profile(mid_file):
    tension_profile = []
    active_notes = set()  # Track currently active notes

    for msg in mid_file.tracks[0]:
        if msg.type == 'note_on' and msg.velocity > 0:
            active_notes.add(msg.note)
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            active_notes.discard(msg.note)

        # Calculate tension score for the current set of active notes
        tension_score = calculate_tension(active_notes)
        tension_profile.append(tension_score)

    return tension_profile

def microseconds_per_quarter_note_to_bpm(microseconds_per_quarter_note):
    return 60000000 / microseconds_per_quarter_note

def tempo_profile(mid_file):
    tempo_changes = []
    current_tempo = 500000  # MIDI default tempo (120 BPM)

    for msg in mid_file.tracks[0]:
        if msg.type == 'set_tempo':
            current_tempo = msg.tempo  # Update current tempo with the new value
            bpm = microseconds_per_quarter_note_to_bpm(current_tempo)
            tempo_changes.append(bpm)  # Convert to BPM and append to the list

    # If no tempo change messages are found, use the default tempo
    if not tempo_changes:
        default_bpm = microseconds_per_quarter_note_to_bpm(current_tempo)
        tempo_changes.append(default_bpm)

    return tempo_changes


#------------------------------------------------------------
# Prepare Output
#------------------------------------------------------------

def plot_profile(combined_profile, title='Combined Intensity Profile'):
    plt.figure(figsize=(10, 6))
    
    # Original intensity profile
    plt.plot(combined_profile, label='Intensity', alpha=0.5)  # Reduced alpha to make spline more visible

    # Spline of best fit
    x = np.arange(len(combined_profile))
    spline = UnivariateSpline(x, combined_profile, s=len(combined_profile))  # s is a smoothing factor
    xs = np.linspace(0, len(combined_profile)-1, 1000)  # More points for a smoother spline curve
    ys = spline(xs)
    plt.plot(xs, ys, label='Spline of Best Fit', color='green')

    # Find the index of the maximum intensity value
    max_intensity_index = combined_profile.index(max(combined_profile))
    total_length = len(combined_profile)
    ratio = max_intensity_index / total_length

    # Draw a vertical line at the index of the maximum intensity
    plt.axvline(x=max_intensity_index, color='r', linestyle='--', label=f'Max Intensity (Ratio: {ratio:.3f})')

    # Annotate the plot with the ratio
    plt.annotate(f'Ratio: {ratio:.3f}', xy=(max_intensity_index, max(combined_profile)),
                 xytext=(max_intensity_index + 0.05 * total_length, max(combined_profile)),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.title(title)
    plt.xlabel('Time (Normalized)')
    plt.ylabel('Intensity (Normalized)')
    plt.legend()
    plt.grid(True)
    plt.show()

    
def normalize_profile(profile):
    if not profile:
        return []
    min_value, max_value = min(profile), max(profile)
    range_value = max_value - min_value
    if range_value == 0:  # Avoid division by zero
        return [0] * len(profile)
    return [(p - min_value) / range_value for p in profile]

def combine_profiles(*profiles, weights=None):
    if weights is None:
        weights = [1] * len(profiles)  # Equal weighting if none specified
    combined_profile = []
    for values in zip(*profiles):
        weighted_sum = sum(value * weight for value, weight in zip(values, weights))
        combined_profile.append(weighted_sum / sum(weights))
    return combined_profile

def build_combined_profile(mid_file, weights=None):
    if weights is None:
        #defaults
        weights = {
            'velocity': 0.25,
            'pitch_density': 0.25,
            'tension': 0.25,
            'tempo': 0.25
        }

    velocity_prof = normalize_profile(velocity_profile(mid_file))
    pitch_density_prof = normalize_profile(pitch_density_profile(mid_file))
    tension_prof = normalize_profile(tension_profile(mid_file))
    tempo_prof = normalize_profile(tempo_profile(mid_file))

    combined_prof = combine_profiles(
        velocity_prof,
        pitch_density_prof,
        tension_prof,
        tempo_prof,
        weights=[weights['velocity'], weights['pitch_density'], weights['tension'], weights['tempo']]
    )
    return combined_prof

def interpolate_profile(profile, target_length):
    x_old = np.linspace(0, 1, len(profile))
    x_new = np.linspace(0, 1, target_length)
    interpolated_profile = np.interp(x_new, x_old, profile)
    return interpolated_profile.tolist()

def average_profiles(profiles):
    if len(profiles) == 1:
        # If there's only one profile, return it as is (or interpolated to a desired length if necessary).
        return profiles[0]
    max_length = max(len(p) for p in profiles)
    interpolated_profiles = [interpolate_profile(p, max_length) for p in profiles]
    averaged_profile = np.mean(interpolated_profiles, axis=0).tolist()
    return averaged_profile

def process_midi_file(midi_file, weights):
    combined_mid = combine_tracks(midi_file)
    combined_prof = build_combined_profile(combined_mid, weights=weights)
    return combined_prof


def get_intensity_profile(midi_file, weights=None):
    if weights is None:
        # Define default weights if none are provided
        weights = {
            'velocity': 0.4, 
            'pitch_density': 0.3, 
            'tension': 0.2, 
            'tempo': 0.1
        }
    combined_mid = combine_tracks(midi_file)
    combined_prof = build_combined_profile(combined_mid, weights=weights)
    return combined_prof

# if __name__ == '__main__':
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     base_dir = os.path.join(current_dir, os.pardir)
#     raw_folder = os.path.join(base_dir, 'raw')
#     midi_files = [os.path.join(raw_folder, f) for f in os.listdir(raw_folder) if f.endswith('.mid')]

#     midi_num = 'all'  # Adjust as needed
#     selected_midi_files = select_midi_files(midi_files, midi_num)

#     all_profiles = []
#     weights = {
#         'velocity': 0.4,  # Weight for velocity profile
#         'pitch_density': 0.3,  # Weight for pitch density profile
#         'tension': 0.20,  # Weight for tension profile
#         'tempo': 0.1  # Weight for tempo profile
#     }

#     # Use ThreadPoolExecutor to process MIDI files in parallel
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         future_to_midi = {executor.submit(process_midi_file, midi_file, weights): midi_file for midi_file in selected_midi_files}
#         for future in concurrent.futures.as_completed(future_to_midi):
#             midi_file = future_to_midi[future]
#             try:
#                 combined_prof = future.result()
#                 all_profiles.append(combined_prof)
#             except Exception as exc:
#                 print(f'{midi_file} generated an exception: {exc}')

#     averaged_profile = average_profiles(all_profiles)
#     plot_profile(averaged_profile, title='Averaged Combined Profile for Selected MIDI Files')

