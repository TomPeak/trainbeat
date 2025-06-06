import pretty_midi
import mido
import pandas as pd
import numpy as np
import os

gm_drum_map = {
    35: "Acoustic Bass Drum",
    36: "Bass Drum 1",
    38: "Acoustic Snare",
    40: "Electric Snare",
    42: "Closed Hi-Hat",
    44: "Pedal Hi-Hat",
    46: "Open Hi-Hat",
    49: "Crash Cymbal 1",
    51: "Ride Cymbal 1",
    # Add more as needed (full GM map: https://midi.org/specs)
}
kick_pitches = [35, 36]
snare_pitches = [38, 40]
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

def parse_midi_four_measures(midi_path, measures_limit=4):
    # Load MIDI
    midi = pretty_midi.PrettyMIDI(midi_path)
    drum_track = None
    for instrument in midi.instruments:
        if instrument.is_drum or (instrument.program == 0 and instrument.channel == 9):  # Channel 10
            drum_track = instrument
            break
    if not drum_track:
        print(f"No drum track in {midi_path}")
        return None, None, None, None

    # Extract time signature
    mido_midi = mido.MidiFile(midi_path)
    time_signature = {'numerator': 4, 'denominator': 4}  # Default 4/4
    for track in mido_midi.tracks:
        for msg in track:
            if msg.type == 'time_signature':
                time_signature = {'numerator': msg.numerator, 'denominator': msg.denominator}
                break
    if time_signature not in [{'numerator': 4, 'denominator': 4}, 
                             {'numerator': 3, 'denominator': 4}, 
                             {'numerator': 6, 'denominator': 8},
                             {'numerator': 5, 'denominator': 8}]:
        print(f"Unsupported time signature {time_signature['numerator']}/{time_signature['denominator']} in {midi_path}")
        return None, None, None, None

    # Calculate measure duration (in seconds)
    bpm = midi.get_tempo_changes()[1][0] if midi.get_tempo_changes()[1].size > 0 else 120
    seconds_per_beat = 60.0 / bpm
    beats_per_measure = time_signature['numerator'] * (4 / time_signature['denominator'])
    measure_duration = beats_per_measure * seconds_per_beat
    max_time = measures_limit * measure_duration

    # Extract kick, snare, and other events within four measures
    kick_events, snare_events, other_events = [], [], []
    for note in drum_track.notes:
        if note.start > max_time:
            continue  # Skip events after four measures
        event = {
            'pitch': note.pitch,
            'drum_name': gm_drum_map.get(note.pitch, f"Unknown_{note.pitch}"),
            'start': note.start,
            'end': note.end,
            'velocity': note.velocity
        }
        if note.pitch in kick_pitches:
            kick_events.append(event)
        elif note.pitch in snare_pitches:
            snare_events.append(event)
        else:
            other_events.append(event)
    
    return time_signature, kick_events, snare_events, other_events

# Load CSV and parse MIDI files
for midi_path in midi_files:
    # Generate prompt from filename (e.g., "funk_fill_89.mid" -> "funk fill")
    filename = os.path.basename(midi_path).replace('.mid', '')
    prompt = filename.split('_')  # e.g., "funk fill" or "rock beat"
    time_sig, kick_events, snare_events, other_events = parse_midi_four_measures(midi_path)
    if kick_events or snare_events or other_events:
        data.append({
            'midi_path': midi_path,
            'time_signature': time_sig,
            'kick_events': kick_events,
            'snare_events': snare_events,
            'other_events': other_events,
            'prompt': prompt[1],
            'bpm':prompt[3],
            'style': prompt[4]
        })
# Create a DataFrame
df = pd.DataFrame(data, columns=['midi_path','time_signature','kick_events','snare_events','other_events','prompt','bpm','style'])

# Step 4: Save the DataFrame to a CSV
output_csv = 'data.csv'  # Output CSV file
df.to_csv(output_csv, index=False)
print(f"CSV saved as {output_csv}")

print("\nProcessing complete")