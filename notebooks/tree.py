import os

def list_dirs(root_dir, exclude_dirs=None, indent=0):
    if exclude_dirs is None:
        exclude_dirs = []

    try:
        entries = os.listdir(root_dir)
    except PermissionError:
        return

    for entry in sorted(entries):
        path = os.path.join(root_dir, entry)
        if os.path.isdir(path):
            print('    ' * indent + '|-- ' + entry)
            if entry not in exclude_dirs:
                list_dirs(path, exclude_dirs, indent + 1)
        else:
            if not any(exclude_dir in root_dir for exclude_dir in exclude_dirs):
                print('    ' * (indent + 1) + '|-- ' + entry)

if __name__ == "__main__":
    root_directory = '.'  # Starting directory
    excluded_directories = ['__pycache__', 'raw', 'processed', 'readable']  # Directories to list but not traverse
    list_dirs(root_directory, excluded_directories)


|-- data
        |-- .DS_Store
        |-- check_npz.py
        |-- check_ranges.py
        |-- midi_to_human_readable.py
        |-- midi_to_preprocess.py
    |-- processed
    |-- raw
    |-- readable
|-- notebooks
|-- outputs
        |-- .DS_Store
    |-- archive
            |-- latest_model_delta log t.h5
            |-- training_log_delta log t.txt
    |-- generated_music
            |-- .DS_Store
            |-- Untitled.aiff
            |-- generated_ensemble_output.midi
            |-- generated_ensemble_output.txt
    |-- models
            |-- .DS_Store
            |-- latest_model.h5
            |-- training_log.txt
    |-- requirements.txt
|-- src
        |-- .DS_Store
    |-- generation
        |-- __pycache__
            |-- generate.py
            |-- midi_conversion.py
            |-- old_generate_function.py
        |-- rules
            |-- __pycache__
                |-- avoid_accumulation.py
                |-- prevent_note_off_without_on.py
                |-- tempo_snap.py
        |-- seeds
            |-- __pycache__
                |-- load_seed.py
                |-- random_seed.py
    |-- models
        |-- __pycache__
            |-- rnn_model.py
    |-- training
            |-- train_model.py