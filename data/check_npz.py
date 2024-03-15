import numpy as np
import os

processed_folder = './processed'
readable_folder = './readable'
file_name = 'beethoven_les_adieux_3_sequences.npz'  # Example file name

# Ensure the 'readable' folder exists, create if not
if not os.path.exists(readable_folder):
    os.makedirs(readable_folder)

output_txt_file = file_name.replace('.npz', '.txt')  # Output text file name
output_txt_path = os.path.join(readable_folder, output_txt_file)  # Output path in 'readable' folder

def npz_to_txt(npz_file, txt_file):
    with np.load(npz_file, allow_pickle=True) as data:
        with open(txt_file, 'w') as f:
            f.write(f"Loaded data from {npz_file}\n\n")

            # Loop through each key in the .npz file and write the data to the text file
            for key in data.keys():
                f.write(f"{key.upper()} data:\n")

                # Handle non-array 'features' data differently
                if key == 'features':
                    features = data[key].item()  # Use .item() to get the dictionary from 0-d array
                    for feature_name, feature_values in features.items():
                        f.write(f"{feature_name}: {feature_values}\n")  # Write all feature values
                else:
                    for sequence in data[key]:  # Iterate through all sequences
                        for line in sequence:  # Write all rows of the sequence
                            f.write(f"{line}\n")
                        f.write("\n")  # Newline for separation between sequences
                f.write("\n")  # Extra newline for separation between different keys

if __name__ == '__main__':
    npz_file_path = os.path.join(processed_folder, file_name)
    npz_to_txt(npz_file_path, output_txt_path)
    print(f"Data from {npz_file_path} has been written to {output_txt_path}")
