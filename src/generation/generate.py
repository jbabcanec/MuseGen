import numpy as np
import random
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import load_model
from midi_conversion import sequence_to_midi


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



def softmax(x, temperature=1.0):
    """Compute softmax values for each set of scores in x adjusted by temperature."""
    e_x = np.exp((x - np.max(x)) / temperature)
    return e_x / e_x.sum(axis=-1, keepdims=True)


def generate_music(models, seed_sequence, num_generate=100, temperature=1.0, max_simultaneous_notes=8):
    input_sequence = np.array(seed_sequence)
    generated_sequence = []

    last_event_time = 0 if len(seed_sequence) == 0 else seed_sequence[-1, -1]
    unresolved_note_ons = set()

    for i in range(num_generate):
        print(f"Generating event {i+1}/{num_generate}")

        for model in models:
            prediction = model.predict(np.expand_dims(input_sequence, axis=0))
            note_event_pred, pitch_pred, velocity_pred, event_time_pred = prediction

            note_event_prob = softmax(note_event_pred[0], temperature=temperature)
            pitch_prob = softmax(pitch_pred[0], temperature=temperature)

            note_event = np.random.choice(range(len(note_event_prob)), p=note_event_prob)
            pitch = np.random.choice(range(len(pitch_prob)), p=pitch_prob)

            best_velocity = velocity_pred[0][0] * 127
            best_event_time = event_time_pred[0].dot(np.arange(event_time_pred.shape[1]))

            # Check note-on accumulation and possibly turn off a note
            if note_event == 1 and len(unresolved_note_ons) >= max_simultaneous_notes:
                note_to_turn_off = unresolved_note_ons.pop()  # Remove and get an unresolved note
                generated_sequence.append([0, note_to_turn_off, 0, best_event_time])  # Add note-off event for it

            if note_event == 1:
                unresolved_note_ons.add(pitch)  # Add new note-on to unresolved
            elif note_event == 0 and pitch in unresolved_note_ons:
                unresolved_note_ons.remove(pitch)  # Remove resolved note-off

            next_event = [note_event, pitch, best_velocity, best_event_time]
            generated_sequence.append(next_event)
            input_sequence = np.vstack([input_sequence[1:], next_event])

            print(f"Selected for event {i+1}: Note Event {note_event}, Pitch {pitch}, Velocity (scaled) {best_velocity:.2f}, Event Time {best_event_time}")
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
