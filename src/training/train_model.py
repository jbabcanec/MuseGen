import sys
import numpy as np
from pathlib import Path
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

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
batch_size = 40  # Adjusted batch size
epochs = 60

# Initialize lists for data accumulation
all_sequences = []
all_pitches = []
all_velocities = []
all_time_deltas = []

# Process each .npz file in the processed data directory
for npz_file in data_processed_dir.glob('*.npz'):
    print(f"Processing file: {npz_file.name}")

    # Load data
    with np.load(npz_file, allow_pickle=True) as data:
        sequences = data['sequences']
        next_events = data['next_events']
    
    # Accumulate data
    all_sequences.append(sequences)
    all_pitches.extend(next_events[:, 1])  # Assuming pitch is the second element
    all_velocities.extend(next_events[:, 2] / 127)  # Normalize velocity
    all_time_deltas.extend(next_events[:, 3])  # Assuming time_delta is the fourth element

# Convert lists to numpy arrays
all_sequences = np.concatenate(all_sequences, axis=0)
all_pitches = np.array(all_pitches)
all_velocities = np.array(all_velocities)
all_time_deltas = np.array(all_time_deltas)

# Determine the range for pitch and time_delta components
pitch_range = int(np.max(all_pitches)) + 1
time_delta_range = int(np.max(all_time_deltas)) + 1

# Prepare data for training
X = all_sequences
y_pitch = to_categorical(all_pitches, num_classes=pitch_range)
y_velocity = all_velocities  # Directly use normalized velocities
y_time_delta = to_categorical(all_time_deltas, num_classes=time_delta_range)

# Define model input shape and build the model
sequence_length, num_features = X.shape[1], X.shape[2]
model = build_multi_output_rnn_model((sequence_length, num_features), num_units=64, dropout_rate=0.3,
                                     pitch_range=pitch_range, velocity_range=1,
                                     time_delta_range=time_delta_range)

# Compile the model with adjusted loss for velocity
model.compile(optimizer='adam', 
              loss={'pitch_output': 'categorical_crossentropy',
                    'velocity_output': 'mean_squared_error',
                    'time_delta_output': 'categorical_crossentropy'},
              metrics=['accuracy'])

# Initialize the EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',      # Monitor the validation loss
    min_delta=0.001,         # Minimum change to qualify as an improvement
    patience=10,             # Number of epochs with no improvement after which training will be stopped
    verbose=1,               # Log when training is stopped
    mode='auto',             # Infer the direction of monitoring (minimizing loss)
    restore_best_weights=True # Restore model weights from the epoch with the best value of the monitored quantity
)

# Train the model
history = model.fit(
    X, {'pitch_output': y_pitch, 'velocity_output': y_velocity, 'time_delta_output': y_time_delta},
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
    callbacks=[early_stopping]  # Include the callback in the training process
)

# Save the trained model
model_save_path = output_models_dir / "combined_model.h5"
model.save(model_save_path)
print(f"Model saved to {model_save_path}")
