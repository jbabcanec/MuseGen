import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import load_model
from midi_conversion import sequence_to_midi

def generate_random_seed(sequence_length=100, pitch_range=128, default_time_delta=120, fourth_component_default=0):
    seed_sequence = []
    for _ in range(sequence_length):
        pitch = np.random.randint(0, pitch_range)  # Keep pitch as MIDI note number
        velocity = np.random.random()  # Generate a random float between 0 and 1 for normalized velocity
        event = [pitch, velocity, default_time_delta, fourth_component_default]
        seed_sequence.append(event)
    return np.array(seed_sequence)


def generate_music(models, seed_sequence, num_generate=500, temperature=1.0):
    input_sequence = np.array(seed_sequence)
    generated_sequence = []
    accumulated_time_delta = 0  # Initialize accumulated time delta

    for i in range(num_generate):
        best_pitch, best_velocity, best_time_delta = 0, 0, 0
        best_pitch_confidence, best_velocity_confidence, best_time_delta_confidence = -np.inf, -np.inf, -np.inf

        for model in models:
            prediction = model.predict(np.expand_dims(input_sequence, axis=0))
            pitch_pred, velocity_pred, time_delta_pred = prediction

            pitch_confidence = np.max(pitch_pred)
            velocity_confidence = np.max(velocity_pred)
            time_delta_confidence = np.max(time_delta_pred)

            if pitch_confidence > best_pitch_confidence:
                best_pitch_confidence = pitch_confidence
                best_pitch = np.argmax(pitch_pred, axis=-1)[0]

            if velocity_confidence > best_velocity_confidence:
                best_velocity_confidence = velocity_confidence
                best_velocity = velocity_pred[0][0]  # Use continuous velocity prediction

            if time_delta_confidence > best_time_delta_confidence:
                best_time_delta_confidence = time_delta_confidence
                best_time_delta = np.argmax(time_delta_pred, axis=-1)[0]  # Assuming time delta is not transformed

        accumulated_time_delta += best_time_delta  # Accumulate time delta for event spacing
        next_event = [best_pitch, best_velocity, accumulated_time_delta, 0]  # Use continuous velocity as is
        generated_sequence.append(next_event)
        input_sequence = np.vstack([input_sequence[1:], next_event])

    return generated_sequence

# Directory setup
project_root = Path(__file__).resolve().parents[2]
models_dir = project_root / 'outputs/models'
generated_music_dir = project_root / 'outputs/generated_music'
generated_music_dir.mkdir(exist_ok=True)

# Load a random seed sequence
seed_sequence = generate_random_seed()

# Load all models into a list
models = [load_model(model_path) for model_path in models_dir.glob('*.h5')]

# Generate a sequence of musical events using the ensemble of models
generated_sequence = generate_music(models, seed_sequence, num_generate=500)

# Convert the generated sequence to MIDI format and save
output_path = generated_music_dir / 'generated_ensemble_output.midi'
sequence_to_midi(generated_sequence, output_path)

print(f"Generated music saved to {output_path}")
with open("generated_sequence.txt", "w") as file:
    file.write(str(generated_sequence))
