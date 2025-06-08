import numpy as np
import random
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import load_model
from midi_conversion import convert_to_midi

# import seeds
from seeds.random_seed import generate_random_seed
from seeds.load_seed import load_random_seed

# import rules which should theoretically disolve
from rules.avoid_accumulation import avoid_accumulation
from rules.prevent_note_off_without_on import prevent_note_off_without_on


# Directory setup
project_root = Path(__file__).resolve().parents[2]
models_dir = project_root / 'outputs/models'
generated_music_dir = project_root / 'outputs/generated_music'
sample_dir = project_root / 'data/processed'
generated_music_dir.mkdir(exist_ok=True)

def softmax(x, temperature):
    """Compute softmax values for each set of scores in x adjusted by temperature."""
    e_x = np.exp((x - np.max(x)) / temperature)
    return e_x / e_x.sum(axis=-1, keepdims=True)


def sample_distribution(probabilities, temperature=1.0, top_k=None, allowed_indices=None):
    """Sample an index from a probability distribution with optional constraints."""
    probs = np.log(probabilities + 1e-8) / temperature
    probs = np.exp(probs)
    probs /= np.sum(probs)
    indices = np.arange(len(probabilities))

    if allowed_indices is not None:
        indices = np.array(allowed_indices)
        probs = probs[indices]
        probs /= np.sum(probs)

    if top_k is not None and top_k < len(indices):
        top_k_idx = np.argsort(probs)[-top_k:]
        indices = indices[top_k_idx]
        probs = probs[top_k_idx]
        probs /= np.sum(probs)

    return np.random.choice(indices, p=probs)


def generate_music(
    models,
    seed_sequence,
    num_generate=100,
    temperature=1.0,
    max_simultaneous_notes=8,
    pitch_low=21,
    pitch_high=108,
    top_k_pitch=8,
):
    input_sequence = np.array(seed_sequence)
    generated_sequence = []

    # Initialize the cumulative event time starting from 0
    cumulative_event_time = 0

    unresolved_note_ons = set()

    for i in range(num_generate):
        print(f"Generating event {i+1}/{num_generate}")

        for model in models:
            prediction = model.predict(np.expand_dims(input_sequence, axis=0))
            note_event_pred, pitch_pred, velocity_pred, event_time_pred = prediction

            note_event_prob = note_event_pred[0]
            pitch_prob = pitch_pred[0]

            note_event = sample_distribution(note_event_prob, temperature=temperature)
            pitch = sample_distribution(
                pitch_prob,
                temperature=temperature,
                top_k=top_k_pitch,
                allowed_indices=range(pitch_low, pitch_high + 1),
            )

            velocity = sample_distribution(velocity_pred[0], temperature=temperature)

            # Calculate the delta time for the event and add it to the cumulative time
            event_time_delta = sample_distribution(event_time_pred[0], temperature=temperature)
            cumulative_event_time += event_time_delta

            unresolved_note_ons, note_off_event = avoid_accumulation(
                unresolved_note_ons,
                note_event,
                pitch,
                max_simultaneous_notes,
                cumulative_event_time,
            )

            if note_off_event:
                generated_sequence.append(note_off_event)  # Add note-off event if returned

            next_event = [note_event, pitch, velocity, cumulative_event_time]
            generated_sequence.append(next_event)
            input_sequence = np.vstack([input_sequence[1:], next_event])

            print(
                f"Selected for event {i+1}: Note Event {note_event}, Pitch {pitch}, Velocity {velocity}, Event Time {cumulative_event_time}"
            )
            print("------")

    # Filter out invalid note-off events
    final_sequence = prevent_note_off_without_on(np.array(generated_sequence))

    return final_sequence

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
generated_sequence = generate_music(
    models,
    seed_sequence,
    num_generate=100,
    temperature=0.9,
    pitch_low=36,
    pitch_high=96,
    top_k_pitch=12,
)

print(generated_sequence)

# Convert the generated sequence to MIDI format and save
output_path = generated_music_dir / 'generated_ensemble_output.midi'
convert_to_midi(generated_sequence, output_path)

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


