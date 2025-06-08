import sys
import argparse
import numpy as np
from pathlib import Path
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

# Adjust Python path for relative imports
current_script_path = Path(__file__).resolve()
src_dir = current_script_path.parent.parent
project_root = src_dir.parent
sys.path.append(str(src_dir))

# Import the multi-output RNN model builder function
from models.rnn_model import build_multi_output_rnn_model
from models.transformer_model import build_transformer_model

# Define paths relative to project root
data_processed_dir = project_root / 'data/processed'
output_models_dir = project_root / 'outputs/models'
log_file_path = output_models_dir / 'training_log.txt'

# Ensure the output directory exists
output_models_dir.mkdir(parents=True, exist_ok=True)

# Define fixed ranges
FIXED_PITCH_RANGE = 128  # MIDI has 128 pitches
FIXED_TIME_DELTA_RANGE = 3500  # Define according to your data
FIXED_VELOCITY_RANGE = 128  # Since velocities are normalized


# Parameters (adjust as needed)
batch_size = 40
epochs = 40
validation_split = 0.2
early_stopping_patience = 10
num_units=64
dropout_rate=0.3

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train music generation model")
parser.add_argument(
    "--model_type",
    choices=["rnn", "transformer"],
    default="rnn",
    help="Choose underlying architecture",
)
args = parser.parse_args()

# Initialize the EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=early_stopping_patience,
    verbose=1,
    mode='auto',
    restore_best_weights=True
)

# Function to process and train model per npz file
def process_and_train(npz_file, model=None):
    print(f"Processing file: {npz_file.name}")

    # Load data
    with np.load(npz_file, allow_pickle=True) as data:
        sequences = data['sequences']  # Adjust key if necessary
        next_events = data['next_events']  # Adjust key if necessary

    # Prepare data for training
    note_events = next_events[:, 0]  # Assuming note event is the first element
    
    invalid_events = [(i, event) for i, event in enumerate(note_events) if event not in [0, 1]]
    if invalid_events:
        print(f"Skipping {npz_file.name} due to invalid note event values:")
        for i, event in invalid_events:
            print(f"  Invalid value {event} at event number {i}")

        with open(log_file_path, 'a') as log_file:
            log_file.write(f"Skipped {npz_file.name} due to invalid note event values: {[event for _, event in invalid_events]}\n")
        return model  # Skip the rest of the function

    pitches = next_events[:, 1]  # Assuming pitch is the second element
    velocities = next_events[:, 2]  # Assuming velocity is the third element
    time_deltas = next_events[:, 3]  # Assuming time_delta is the fourth element

    X = sequences
    y_note_event = to_categorical(note_events, num_classes=2)  # Note on/off
    y_pitch = to_categorical(pitches, num_classes=FIXED_PITCH_RANGE)
    y_velocity =to_categorical(velocities, num_classes=FIXED_VELOCITY_RANGE)
    y_time_delta = to_categorical(time_deltas, num_classes=FIXED_TIME_DELTA_RANGE)

    if model is None:
        print("Initializing new model.")
        sequence_length, num_features = X.shape[1], X.shape[2]
        if args.model_type == "transformer":
            model = build_transformer_model(
                (sequence_length, num_features),
                pitch_range=FIXED_PITCH_RANGE,
                velocity_range=FIXED_VELOCITY_RANGE,
                time_delta_range=FIXED_TIME_DELTA_RANGE,
            )
        else:
            model = build_multi_output_rnn_model(
                (sequence_length, num_features),
                num_units=num_units,
                dropout_rate=dropout_rate,
                pitch_range=FIXED_PITCH_RANGE,
                velocity_range=FIXED_VELOCITY_RANGE,
                time_delta_range=FIXED_TIME_DELTA_RANGE,
            )

        model.compile(optimizer='adam',
                      loss={'note_event_output': 'categorical_crossentropy',
                            'pitch_output': 'categorical_crossentropy',
                            'velocity_output': 'categorical_crossentropy',
                            'time_delta_output': 'categorical_crossentropy'},
                      metrics={'note_event_output': 'accuracy',
                               'pitch_output': 'accuracy',
                               'velocity_output': 'accuracy',
                               'time_delta_output': 'accuracy'})

    # Train the model
    model.fit(X, {'note_event_output': y_note_event, 'pitch_output': y_pitch, 'velocity_output': y_velocity, 'time_delta_output': y_time_delta},
              batch_size=batch_size, epochs=epochs, validation_split=validation_split, callbacks=[early_stopping])

    print(f"Completed processing {npz_file.name}")

    # Log the processed file
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"{npz_file.name}\n")

    return model


# Read the existing training log
def read_training_log(log_file_path):
    processed_files = set()
    if log_file_path.exists():
        with open(log_file_path, 'r') as log_file:
            processed_files = {line.strip() for line in log_file}
    return processed_files

processed_files = read_training_log(log_file_path)

# Initialize model to None, it will be created or loaded if exists
model = None
model_path = output_models_dir / "latest_model.h5"

# Check if a previously trained model exists and load it
if model_path.exists():
    print(f"Loading model from {model_path}")
    model = load_model(model_path)

# Process each .npz file in the processed data directory
for npz_file in sorted(data_processed_dir.glob('*.npz')):
    if npz_file.name in processed_files:
        print(f"Skipping already processed file: {npz_file.name}")
        continue

    # Process and train the model with the current .npz file
    model = process_and_train(npz_file, model)

    # Save the latest model after each file is processed to facilitate resuming training
    latest_model_save_path = output_models_dir / "latest_model.h5"
    model.save(latest_model_save_path)
    print(f"Latest model saved to {latest_model_save_path}")
