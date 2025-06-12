# Music Generation Project

This project aims to generate music using a deep learning model trained on MIDI files. It involves processing MIDI files, training a recurrent neural network (RNN) model, and using the trained model to generate new music sequences, which are then converted back into MIDI format.

## Overview

The project is structured into several key scripts:

- `midi_to_preprocess.py`: Converts raw MIDI files into a processed format suitable for training, including encoding of MIDI events and creating training sequences.
- `augmentation.py`: Provides transposition, time stretching, velocity scaling, and optional intensity based augmentation to artificially increase the training set
- `features.py`: Extracts features and appends them to .npz files. These features include harmony and intensity profile. More may be added.
- `validations/midi_to_human_readable.py`: Converts MIDI files into a human-readable format for debugging and understanding the MIDI data structure.
- `validations/check_npz.py`: Converts chosen .npz file into text files for debugging.
- `validations/intensity_profile.py`: Examines intensity of piece(s) as a function of velocity, tempo, tension, and pitch aggregation. For research purposes.
- `validations/dataset_statistics.py`: Aggregates pitch, velocity, and time delta distributions across processed files to guide further modeling research.
- `train_model.py`: Trains either an RNN or Transformer model on the processed MIDI data, capable of predicting pitch, velocity, and time delta for the next event in a sequence.
- `rnn_model.py`: Defines the GRU-based architecture.
- `transformer_model.py`: Provides a Transformer architecture for advanced training.
- `generate.py`: Uses the trained model(s) to generate new music sequences based on a random seed or a given input sequence.
- `rules/...`: These are "dissolvable" rules which aid music generation. They are dissolvable because they will eventually not be invoked as training improves.
- `seeds/...`: These are different starting sequences that can be invoked.
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

3. Run additional preprocessing such as `augmentation.py` and `features.py`.
   The augmentation script now supports velocity scaling and applying the
   intensity profile of the original MIDI when available:

    ```
    python augmentation.py
    python features.py
    ```

### Training the Model

1. After preprocessing, use `train_model.py` to train a model. The script now
   supports an optional Transformer architecture:

    ```
    # Train GRU-based model
    python train_model.py

    # Train Transformer model
    python train_model.py --model_type transformer
    ```

2. Trained models are saved in the `./outputs/models` directory.

### Generating Music

Run `generate.py` to create new music using your trained models. The script now
supports command-line options for customizing generation:

```bash
python src/generation/generate.py --num-generate 200 --temperature 1.0 \
    --output outputs/generated_music/my_song.midi
```

Generated MIDI files are saved to `./outputs/generated_music` by default.

### Converting MIDI to Human-Readable Format

1. Run `midi_to_human_readable.py` to convert MIDI files into a human-readable text format for debugging:

    ```
    python midi_to_human_readable.py
    ```

## Customization

- You can adjust model parameters, sequence length, and other configurations by modifying the respective scripts.
- The generation script (`generate.py`) now accepts command-line arguments so you can tweak the seed source, number of generated events, and other parameters without modifying the code.

## License

[MIT License](LICENSE)
