# Music Generation Project

This project aims to generate music using a deep learning model trained on MIDI files. It involves processing MIDI files, training a recurrent neural network (RNN) model, and using the trained model to generate new music sequences, which are then converted back into MIDI format.

## Overview

The project is structured into several key scripts:

- `midi_to_preprocess.py`: Converts raw MIDI files into a processed format suitable for training, including encoding of MIDI events and creating training sequences.
- `midi_to_human_readable.py`: Converts MIDI files into a human-readable format for debugging and understanding the MIDI data structure.
- `train_model.py`: Trains a multi-output RNN model on the processed MIDI data, capable of predicting pitch, velocity, and time delta for the next event in a sequence.
- `rnn_model.py`: Defines the RNN model architecture used for training in `train_model.py`.
- `generate.py`: Uses the trained model(s) to generate new music sequences based on a random seed or a given input sequence.
- `midi_conversion.py`: Converts the generated music sequences back into MIDI format, ready for playback.

## Prerequisites

- Python 3.6 or newer
- TensorFlow 2.x
- NumPy
- mido
- pretty_midi (for `midi_to_human_readable.py` only)

## Installation

1. Clone this repository to your local machine.
2. Install the required Python packages by running:

    ```
    pip install -r requirements.txt
    ```

## Usage

### Preprocessing MIDI Files

1. Place raw MIDI files in the `./raw` directory.
2. Run `midi_to_preprocess.py` to convert raw MIDI files into processed sequences:

    ```
    python midi_to_preprocess.py
    ```

### Training the Model

1. After preprocessing, use `train_model.py` to train the RNN model:

    ```
    python train_model.py
    ```

2. Trained models are saved in the `./outputs/models` directory.

### Generating Music

1. Use `generate.py` to generate new music sequences with the trained model:

    ```
    python generate.py
    ```

2. Generated MIDI files are saved in the `./outputs/generated_music` directory.

### Converting MIDI to Human-Readable Format

1. Run `midi_to_human_readable.py` to convert MIDI files into a human-readable text format for debugging:

    ```
    python midi_to_human_readable.py
    ```

## Customization

- You can adjust model parameters, sequence length, and other configurations by modifying the respective scripts.
- The generation script (`generate.py`) allows for customization of the seed sequence, number of generated events, and ensemble model usage.

## License

[MIT License](LICENSE)
