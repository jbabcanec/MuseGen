import mido
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor

raw_folder = './raw'
processed_folder = './processed'
sequence_length = 100  # Define the sequence length
velocity_threshold = 10  # Minimum velocity to consider a note-on event as an actual note
epsilon = 1e-5  # Small value to avoid log(0)

if not os.path.exists(processed_folder):
    os.makedirs(processed_folder)

def encode_event(event_type, pitch, velocity, delta_log_time=None):
    encoded_event_type = 0 if event_type == 'OFF' else 1 if velocity > velocity_threshold else 2
    # Only include delta_log_time in the encoded event
    return [
        encoded_event_type,
        pitch,
        velocity / 127,
        delta_log_time if delta_log_time is not None else 0  # Use delta_log_time directly
    ]

def log_event_distribution(events):
    event_counts = {'OFF': 0, 'ON_above_thresh': 0, 'ON_below_thresh': 0}
    for event in events:
        event_type = event[0]
        if event_type == 0:
            event_counts['OFF'] += 1
        elif event_type == 1:
            event_counts['ON_above_thresh'] += 1
        else:
            event_counts['ON_below_thresh'] += 1
    print(f"Event distribution - OFF: {event_counts['OFF']}, ON (above threshold): {event_counts['ON_above_thresh']}, ON (below threshold): {event_counts['ON_below_thresh']}")

def process_file(midi_file):
    print("Processing file:", midi_file)
    path = os.path.join(raw_folder, midi_file)
    try:
        mid = mido.MidiFile(path)
    except Exception as e:
        print(f"Error reading {midi_file}: {e}")
        return

    events = []

    for track in mid.tracks:
        current_time = 0  # Reset current time for each track
        for msg in track:
            current_time += msg.time  # Accumulate time
            if msg.type in ['note_on', 'note_off']:
                event_type = 1 if msg.type == 'note_on' and msg.velocity > 0 else 0  # Use 1 for 'ON', 0 for 'OFF'
                events.append((current_time, event_type, msg.note, msg.velocity))  # Include event_type as a numerical value

    # Sort events first by time and then by event type ('OFF' before 'ON')
    events.sort(key=lambda x: (x[0], x[1]))

    # Calculate logged times and their differences
    log_times = [np.log1p(event[0] + epsilon) for event in events]
    delta_log_times = [log_times[i] - log_times[i-1] for i in range(1, len(log_times))]
    delta_log_times.insert(0, 0)  # First event has no previous event

    # Encode events using only delta_log_times and log their distribution
    encoded_events = [encode_event('ON' if etype == 1 else 'OFF', note, velocity, delta_log_times[i]) for i, (_, etype, note, velocity) in enumerate(events)]
    log_event_distribution(encoded_events)

    # Generate sequences and their next events
    sequences, next_events = zip(*[(encoded_events[i:i+sequence_length], encoded_events[i+sequence_length]) for i in range(len(encoded_events) - sequence_length)])

    sequences = np.array(sequences)
    next_events = np.array(next_events)

    # Save sequences and their next events to a compressed file
    output_file = os.path.join(processed_folder, midi_file.replace('.mid', '').replace('.midi', '') + '_sequences.npz')
    np.savez_compressed(output_file, sequences=sequences, next_events=next_events)
    print(f"Saved {len(sequences)} sequences and their next events to {output_file}.\n")

if __name__ == '__main__':
    midi_files = [f for f in os.listdir(raw_folder) if f.endswith('.mid') or f.endswith('.midi')]
    with ProcessPoolExecutor() as executor:
        executor.map(process_file, midi_files)
