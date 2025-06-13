import sys
import argparse
import numpy as np
from pathlib import Path
import tensorflow as tf
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

def load_file(npz_path):
    """Return sequences and next events from an ``npz`` file."""
    with np.load(npz_path, allow_pickle=True) as data:
        return data['sequences'], data['next_events']


def build_dataset(npz_files):
    """Create a ``tf.data.Dataset`` streaming samples from all ``npz`` files."""
    file_contents = []
    sample_shape = None
    total_samples = 0

    for path in npz_files:
        seq, nxt = load_file(path)
        if sample_shape is None:
            sample_shape = seq.shape[1:]
        file_contents.append((seq, nxt))
        total_samples += seq.shape[0]

    def generator():
        for seq, nxt in file_contents:
            for s, n in zip(seq, nxt):
                yield s.astype(np.float32), n.astype(np.int32)

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=sample_shape, dtype=tf.float32),
            tf.TensorSpec(shape=(4,), dtype=tf.int32),
        ),
    )

    dataset = dataset.map(
        lambda seq, evt: (
            seq,
            {
                'note_event_output': tf.one_hot(evt[0], depth=2),
                'pitch_output': tf.one_hot(evt[1], depth=FIXED_PITCH_RANGE),
                'velocity_output': tf.one_hot(evt[2], depth=FIXED_VELOCITY_RANGE),
                'time_delta_output': tf.one_hot(evt[3], depth=FIXED_TIME_DELTA_RANGE),
            },
        )
    )

    return dataset.cache(), sample_shape, total_samples


# Read the existing training log
def read_training_log(log_file_path):
    processed_files = set()
    if log_file_path.exists():
        with open(log_file_path, 'r') as log_file:
            processed_files = {line.strip() for line in log_file}
    return processed_files

processed_files = read_training_log(log_file_path)

npz_files = [f for f in sorted(data_processed_dir.glob('*.npz')) if f.name not in processed_files]

if not npz_files:
    raise ValueError(f"No new .npz files found in {data_processed_dir}")

dataset, sample_shape, total_samples = build_dataset(npz_files)

val_size = int(total_samples * validation_split)
val_dataset = dataset.take(val_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
train_dataset = dataset.skip(val_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

sequence_length, num_features = sample_shape

model = None
model_path = output_models_dir / "latest_model.h5"
if model_path.exists():
    print(f"Loading model from {model_path}")
    model = load_model(model_path)

if model is None:
    print("Initializing new model.")
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

    model.compile(
        optimizer="adam",
        loss={
            "note_event_output": "categorical_crossentropy",
            "pitch_output": "categorical_crossentropy",
            "velocity_output": "categorical_crossentropy",
            "time_delta_output": "categorical_crossentropy",
        },
        metrics={
            "note_event_output": "accuracy",
            "pitch_output": "accuracy",
            "velocity_output": "accuracy",
            "time_delta_output": "accuracy",
        },
    )

model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks=[early_stopping],
)

model.save(model_path)
print(f"Model saved to {model_path}")

# Update training log with processed files
with open(log_file_path, "a") as log_file:
    for f in npz_files:
        log_file.write(f"{f.name}\n")
