import numpy as np
import os

# Get the directory where this script is located
current_dir = os.path.dirname(__file__)

# Go up one directory from the current script's location
base_dir = os.path.join(current_dir, os.pardir)

# Define the processed and readable folders relative to the base directory
processed_folder = os.path.join(base_dir, 'processed')
readable_folder = os.path.join(base_dir, 'readable')

file_name = 'beethoven_les_adieux_3_sequences.npz'  # Example file name
choice = 'features'  # The section you want to extract

# Ensure the 'readable' folder exists, create if not
if not os.path.exists(readable_folder):
    os.makedirs(readable_folder)

output_txt_file = file_name.replace('.npz', '.txt')  # Output text file name
output_txt_path = os.path.join(readable_folder, output_txt_file)  # Output path in 'readable' folder

def npz_to_txt(npz_file, txt_file, section=None):
    with np.load(npz_file, allow_pickle=True) as data:
        with open(txt_file, 'w') as f:
            f.write(f"Loaded data from {npz_file}\n\n")
            
            if section and section in data.keys():
                f.write(f"{section.upper()} data:\n")
                if section == 'features':
                    features = data[section].item()  # Use .item() to get the dictionary from 0-d array
                    for feature_name, feature_values in features.items():
                        f.write(f"{feature_name}: {feature_values}\n")
                else:
                    for sequence in data[section]:
                        for line in sequence:
                            f.write(f"{line}\n")
                        f.write("\n")
            elif section:
                f.write(f"Section '{section}' not found in the data.\n")
            else:
                # If no specific section is provided, print all data (original behavior)
                for key in data.keys():
                    f.write(f"{key.upper()} data:\n")
                    if key == 'features':
                        features = data[key].item()
                        for feature_name, feature_values in features.items():
                            f.write(f"{feature_name}: {feature_values}\n")
                    else:
                        for sequence in data[key]:
                            for line in sequence:
                                f.write(f"{line}\n")
                            f.write("\n")
                    f.write("\n")

if __name__ == '__main__':
    npz_file_path = os.path.join(processed_folder, file_name)
    npz_to_txt(npz_file_path, output_txt_path, choice)
    print(f"Data from {npz_file_path} has been written to {output_txt_path}")
