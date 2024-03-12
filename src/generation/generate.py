import numpy as np
import random
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import load_model
from midi_conversion import sequence_to_midi

def generate_random_seed(sequence_length=100, most_likely_first_pitch=60, pitch_range=128, time_increment=0.1):
    current_time = 0  # Initialize the current time
    seed_sequence = []

    # Start with an initial event with the most likely pitch, full velocity, and initial time
    seed_sequence.append([1, most_likely_first_pitch, 1.0, current_time])

    # Generate subsequent events with incremented time values
    for _ in range(1, sequence_length):
        pitch = np.random.randint(0, pitch_range)  # Random pitch within the range
        velocity = np.random.random()  # Random velocity between 0 and 1
        current_time += time_increment  # Increment the current time

        # Generate a new event with a note-on, random pitch, random velocity, and updated current time
        event = [1, pitch, velocity, current_time]
        seed_sequence.append(event)

    return np.array(seed_sequence)

# Function to load a random seed sequence from a .npz file in the sample_dir
def load_random_seed(sample_dir, sequence_length=100):
    npz_files = list(Path(sample_dir).glob('*.npz'))

    if not npz_files:
        raise ValueError("No .npz files found in the specified directory.")

    random_npz = random.choice(npz_files)
    print(f"Loading seed sequence from: {random_npz}")

    with np.load(random_npz) as data:
        # Assuming 'sequences' is a 3D array ([num_sequences, sequence_length, num_features])
        # and you want to select one sequence (2D) from it
        all_sequences = data['sequences']

        # Select a single sequence randomly or use a specific selection criteria
        selected_sequence_index = random.randint(0, all_sequences.shape[0] - 1)
        seed_sequence = all_sequences[selected_sequence_index]

        # Adjust the sequence to have the desired length
        if seed_sequence.shape[0] > sequence_length:
            # Truncate the sequence if it's longer than the desired length
            seed_sequence = seed_sequence[:sequence_length, :]
        elif seed_sequence.shape[0] < sequence_length:
            # Pad the sequence with zeros if it's shorter than the desired length
            padding = np.zeros((sequence_length - seed_sequence.shape[0], seed_sequence.shape[1]))
            seed_sequence = np.vstack([seed_sequence, padding])

    return seed_sequence


def generate_music(models, seed_sequence, num_generate=100, temperature=1.0):
    input_sequence = np.array(seed_sequence)
    generated_sequence = []

    for i in range(num_generate):
        best_pitch, best_velocity, best_event_time, best_note_event = 0, 0, 0, 0
        best_pitch_confidence, best_velocity_confidence, best_event_time_confidence, best_note_event_confidence = -np.inf, -np.inf, -np.inf, -np.inf

        print(f"Generating event {i+1}/{num_generate}")

        for model in models:
            prediction = model.predict(np.expand_dims(input_sequence, axis=0))
            note_event_pred, pitch_pred, velocity_pred, event_time_pred = prediction

            note_event_confidence = np.max(note_event_pred)
            pitch_confidence = np.max(pitch_pred)
            velocity_confidence = np.max(velocity_pred)
            event_time_confidence = np.max(event_time_pred)

            if note_event_confidence > best_note_event_confidence:
                best_note_event_confidence = note_event_confidence
                best_note_event = np.argmax(note_event_pred, axis=-1)[0]

            if pitch_confidence > best_pitch_confidence:
                best_pitch_confidence = pitch_confidence
                best_pitch = np.argmax(pitch_pred, axis=-1)[0]

            if velocity_confidence > best_velocity_confidence:
                best_velocity_confidence = velocity_confidence
                best_velocity = velocity_pred[0][0] * 127

            if event_time_confidence > best_event_time_confidence:
                best_event_time_confidence = event_time_confidence
                best_event_time = event_time_pred[0].dot(np.arange(event_time_pred.shape[1]))  # Expected value for event time

            print(f"  Model: {model.name} - Note Event: {best_note_event} (Confidence: {best_note_event_confidence:.4f}), Pitch: {best_pitch} (Confidence: {best_pitch_confidence:.4f}), Velocity: {best_velocity:.2f} (Confidence: {best_velocity_confidence:.4f}), Event Time: {best_event_time} (Confidence: {best_event_time_confidence:.4f})")

        next_event = [best_note_event, best_pitch, best_velocity, best_event_time]
        generated_sequence.append(next_event)
        input_sequence = np.vstack([input_sequence[1:], next_event])

        print(f"Selected for event {i+1}: Note Event {best_note_event}, Pitch {best_pitch}, Velocity (scaled) {best_velocity:.2f}, Event Time {best_event_time}")
        print("------")

    return generated_sequence



# Directory setup
project_root = Path(__file__).resolve().parents[2]
models_dir = project_root / 'outputs/models'
generated_music_dir = project_root / 'outputs/generated_music'
sample_dir = project_root / 'data/processed'
generated_music_dir.mkdir(exist_ok=True)



# Initialize the seed sequence with randomness
# -----------------------------------------------------------------------------
# most_likely_first_pitch = 60  # Replace with the actual value determined from your dataset
# seed_sequence = generate_random_seed(most_likely_first_pitch=most_likely_first_pitch)

# Initialize the seed sequence with sample from processed .npz file
# -----------------------------------------------------------------------------
seed_sequence = load_random_seed(sample_dir)

print(seed_sequence)



# Load all models into a list
models = [load_model(model_path) for model_path in models_dir.glob('*.h5')]

# Generate a sequence of musical events using the ensemble of models
generated_sequence = generate_music(models, seed_sequence, num_generate=100)

# Convert the generated sequence to MIDI format and save
output_path = generated_music_dir / 'generated_ensemble_output.midi'
#sequence_to_midi(generated_sequence, output_path)

print(f"Generated music saved to {output_path}")

output_text_path = output_path.with_suffix('.txt')  # Change the extension of the MIDI file path to .txt for the log file

with open(output_text_path, "w") as file:
    file.write("[[")  # Start of the outer list
    
    for i, event in enumerate(generated_sequence):
        note_event, pitch, normalized_velocity, current_time = event
        velocity = normalized_velocity  # Rescale the normalized velocity back to MIDI standards
        
        # Format the event as a list and remove the brackets to match your desired format
        event_str = f"  [{note_event:.0f} {pitch:.0f} {velocity:.8f} {current_time:.8f}]"
        
        # Add a comma after each event except the last one
        if i < len(generated_sequence) - 1:
            event_str += ","
        
        file.write(event_str)
        
        # Add a newline every 4 events for better readability, similar to your example
        if (i + 1) % 4 == 0:
            file.write("\n")
    
    file.write("]]")  # End of the outer list

print(f"Generated sequence details saved to {output_text_path}")
