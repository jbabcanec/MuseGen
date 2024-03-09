import sys
import numpy as np
from pathlib import Path
from tensorflow.keras.utils import to_categorical

# Adjust Python path for relative imports
current_script_path = Path(__file__).resolve()
src_dir = current_script_path.parent.parent
project_root = src_dir.parent
sys.path.append(str(src_dir))

# Import the multi-output RNN model builder function
from models.rnn_model import build_multi_output_rnn_model

# Define paths relative to project root
data_processed_dir = project_root / 'data/processed'
output_models_dir = project_root / 'outputs/models'

# Ensure the output directory exists
output_models_dir.mkdir(parents=True, exist_ok=True)

# Parameters (adjust as needed)
batch_size = 64
epochs = 40

# Process each .npz file in the processed data directory
for npz_file in data_processed_dir.glob('*.npz'):
    print(f"Processing file: {npz_file.name}")

    # Load data
    with np.load(npz_file, allow_pickle=True) as data:
        sequences = data['sequences']
        next_events = data['next_events']

    # Split next_events into separate components
    pitches = next_events[:, 1]  # Assuming pitch is the second element
    velocities = next_events[:, 2] / 127  # Normalize velocity assuming max MIDI velocity is 127
    time_deltas = next_events[:, 3]  # Assuming time_delta is the fourth element

    # Debugging: Check the diversity of pitch values
    unique_pitches = np.unique(pitches)
    print(f"Unique pitches in {npz_file.name}: {unique_pitches}")
    print(f"Number of unique pitches: {len(unique_pitches)}")

    # Determine the range for pitch and time_delta components
    pitch_range = int(np.max(pitches)) + 1
    time_delta_range = int(np.max(time_deltas)) + 1

    # Prepare data for training
    X = sequences
    y_pitch = to_categorical(pitches, num_classes=pitch_range)
    y_velocity = velocities  # Directly use normalized velocities
    y_time_delta = to_categorical(time_deltas, num_classes=time_delta_range)

    # Debugging: Check the encoding of pitch values
    print(f"y_pitch shape: {y_pitch.shape}")
    print(f"Example encoded pitch values: {y_pitch[:5]}")

    # Define model input shape and build the model
    sequence_length, num_features = X.shape[1], X.shape[2]
    model = build_multi_output_rnn_model((sequence_length, num_features), num_units=64, dropout_rate=0.2,
                                         pitch_range=pitch_range, velocity_range=1,  # velocity_range is 1 since it's now a continuous output
                                         time_delta_range=time_delta_range)

    # Compile the model with adjusted loss for velocity
    model.compile(optimizer='adam', 
                  loss={'pitch_output': 'categorical_crossentropy',
                        'velocity_output': 'mean_squared_error',  # Changed to MSE for continuous output
                        'time_delta_output': 'categorical_crossentropy'},
                  metrics=['accuracy'])

    # Train the model
    model.fit(X, {'pitch_output': y_pitch, 'velocity_output': y_velocity, 'time_delta_output': y_time_delta},
              batch_size=batch_size, epochs=epochs, validation_split=0.2)

    # Save the trained model
    model_name = f"model_{npz_file.stem}.h5"
    model_save_path = output_models_dir / model_name
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")