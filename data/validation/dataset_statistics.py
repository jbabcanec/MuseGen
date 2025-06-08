import os
import numpy as np
import matplotlib.pyplot as plt

# Determine paths relative to this script
current_dir = os.path.dirname(__file__)
base_dir = os.path.join(current_dir, os.pardir)
processed_folder = os.path.join(base_dir, 'processed')
results_folder = os.path.join(current_dir, 'results')

os.makedirs(results_folder, exist_ok=True)

# Initialize containers for statistics
pitch_counts = np.zeros(128, dtype=int)
velocity_counts = np.zeros(128, dtype=int)
seq_lengths = []
time_delta_counts = {}

note_count = 0
cumulative_time = 0

for file_name in os.listdir(processed_folder):
    if not file_name.endswith('.npz'):
        continue
    npz_path = os.path.join(processed_folder, file_name)
    with np.load(npz_path, allow_pickle=True) as data:
        for key in data.keys():
            if 'sequences' in key:
                sequences = data[key]
                for seq in sequences:
                    seq_lengths.append(len(seq))
                    for event in seq:
                        event_type, pitch, velocity, delta_t = event
                        pitch_counts[pitch] += 1
                        velocity_counts[velocity] += 1
                        time_delta_counts[delta_t] = time_delta_counts.get(delta_t, 0) + 1
                        note_count += 1
                        cumulative_time += delta_t

# Compute aggregate metrics
average_seq_len = np.mean(seq_lengths) if seq_lengths else 0
avg_time_delta = cumulative_time / note_count if note_count else 0

print(f"Processed {len(seq_lengths)} sequences")
print(f"Average sequence length: {average_seq_len:.2f} events")
print(f"Average time delta: {avg_time_delta:.2f}")

# Plot pitch distribution
plt.figure(figsize=(10,4))
plt.bar(range(128), pitch_counts, color='skyblue')
plt.title('Pitch Distribution')
plt.xlabel('MIDI Pitch')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(results_folder, 'pitch_distribution.png'))

# Plot velocity distribution
plt.figure(figsize=(10,4))
plt.bar(range(128), velocity_counts, color='salmon')
plt.title('Velocity Distribution')
plt.xlabel('Velocity')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(results_folder, 'velocity_distribution.png'))

# Plot time delta histogram
sorted_deltas = sorted(time_delta_counts.items())
if sorted_deltas:
    deltas, counts = zip(*sorted_deltas)
    plt.figure(figsize=(10,4))
    plt.bar(deltas, counts, color='gray')
    plt.title('Time Delta Distribution')
    plt.xlabel('Time Delta')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, 'time_delta_distribution.png'))
