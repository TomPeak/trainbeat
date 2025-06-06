import pandas as pd
import os
from pathlib import Path

# Step 1: Define the root folder
groove_folder = './Groove'  # Replace with your Groove folder path if different

# Step 2: Recursively find all .mid files
midi_files = []
for root, _, files in os.walk(groove_folder):
    for file in files:
        if file.endswith('.mid'):
            midi_files.append(os.path.join(root, file))

if not midi_files:
    print(f"No MIDI files found in {groove_folder}")
    exit()

# Step 3: Generate a dat.csv-like structure
data = []
for midi_path in midi_files:
    # Generate prompt from filename (e.g., "funk_fill_89.mid" -> "funk fill")
    filename = os.path.basename(midi_path).replace('.mid', '')
    prompt = filename.split('_')  # e.g., "funk fill" or "rock beat"
    data.append({'midi_file': midi_path, 'prompt': prompt[1], 'bpm': prompt[2], 'style': prompt[3]})

# Create a DataFrame
df = pd.DataFrame(data, columns=['midi_file', 'prompt','bpm','style'])

# Step 4: Save the DataFrame to a CSV
output_csv = 'data.csv'  # Output CSV file
df.to_csv(output_csv, index=False)
print(f"CSV saved as {output_csv}")

print("\nProcessing complete")