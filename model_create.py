#!/usr/bin/env python3
import argparse
import pretty_midi
import mido
import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import random

# GM drum mapping
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
}

groove_folder = './Groove'

midi_files = [os.path.join(root, file) for root, _, files in os.walk(groove_folder) for file in files if file.endswith('.mid')]
if not midi_files:
    print(f"No MIDI files found in {groove_folder}")
    exit()

def parse_midi_four_measures(midi_path, measures_limit=4):
    filename = os.path.basename(midi_path).replace('.mid', '')
    parts = filename.split('_')
    if len(parts) != 5:
        print(f"Invalid filename format: {midi_path}")
        return None, None, None, None
    try:
        prompt = parts[1]
        bpm = float(parts[2])
        style = parts[3]
        time_sig_str = parts[4].split('-')
        time_signature = {'numerator': int(time_sig_str[0]), 'denominator': int(time_sig_str[1])}
    except (IndexError, ValueError):
        print(f"Error parsing filename: {midi_path}")
        return None, None, None, None

    valid_time_sigs = [{'numerator': 4, 'denominator': 4}, 
                       {'numerator': 3, 'denominator': 4}, 
                       {'numerator': 6, 'denominator': 8},
                       {'numerator': 5, 'denominator': 8},
                       {'numerator': 5, 'denominator': 4}]
    if time_signature not in valid_time_sigs:
        print(f"Unsupported time signature {time_signature['numerator']}/{time_signature['denominator']} in {midi_path}")
        return None, None, None, None

    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"Error loading MIDI {midi_path}: {e}")
        return None, None, None, None

    drum_track = None
    for instrument in midi.instruments:
        if instrument.is_drum or (instrument.program == 0 and instrument.channel == 9):
            drum_track = instrument
            break
    if not drum_track:
        print(f"No drum track in {midi_path}")
        return None, None, None, None

    mido_midi = mido.MidiFile(midi_path)
    midi_time_sig = {'numerator': 4, 'denominator': 4}
    for track in mido_midi.tracks:
        for msg in track:
            if msg.type == 'time_signature':
                midi_time_sig = {'numerator': msg.numerator, 'denominator': msg.denominator}
                break
    if midi_time_sig != time_signature:
        print(f"Warning: MIDI time signature {midi_time_sig} differs from filename {time_signature} in {midi_path}")
        time_signature = midi_time_sig

    seconds_per_beat = 60.0 / bpm
    beats_per_measure = time_signature['numerator'] * (4 / time_signature['denominator'])
    measure_duration = beats_per_measure * seconds_per_beat
    max_time = measures_limit * measure_duration

    drum_events = []
    for note in drum_track.notes:
        if note.start > max_time:
            continue
        drum_events.append({
            'pitch': note.pitch,
            'drum_name': gm_drum_map.get(note.pitch, f"Unknown_{note.pitch}"),
            'start': note.start,
            'end': note.end,
            'velocity': note.velocity
        })

    return time_signature, drum_events, bpm, {'prompt': prompt, 'style': style}

midi_data = []
for midi_path in midi_files:
    time_sig, drum_events, bpm, metadata = parse_midi_four_measures(midi_path)
    if drum_events:
        midi_data.append({
            'midi_path': midi_path,
            'time_signature': time_sig,
            'drum_events': drum_events,
            'bpm': bpm,
            'prompt': metadata['prompt'],
            'style': metadata['style']
        })

if not midi_data:
    print("No valid MIDI files processed")
    exit()

prompt_encoder = LabelEncoder()
style_encoder = LabelEncoder()
time_sig_encoder = LabelEncoder()

prompts = [d['prompt'] for d in midi_data]
styles = [d['style'] for d in midi_data]
time_sigs = [f"{d['time_signature']['numerator']}/{d['time_signature']['denominator']}" for d in midi_data]

prompt_encoded = prompt_encoder.fit_transform(prompts)
style_encoded = style_encoder.fit_transform(styles)
time_sig_encoded = time_sig_encoder.fit_transform(time_sigs)

for i, data in enumerate(midi_data):
    data['prompt_encoded'] = prompt_encoded[i]
    data['style_encoded'] = style_encoded[i]
    data['time_sig_encoded'] = time_sig_encoded[i]
    data['text_input'] = [data['bpm'], data['style_encoded'], data['prompt_encoded'], data['time_sig_encoded']]

def tokenize_midi_events(events, bpm, time_signature, resolution=32):
    seconds_per_beat = 60.0 / bpm
    beats_per_measure = time_signature['numerator'] * (4 / time_signature['denominator'])
    seconds_per_tick = seconds_per_beat / (resolution / 4)
    tokens = []
    for event in events:
        time_step = int(round(event['start'] / seconds_per_tick))
        measure = time_step // (beats_per_measure * (resolution // time_signature['denominator']))
        beat_in_measure = (time_step % (beats_per_measure * (resolution // time_signature['denominator']))) / (resolution // time_signature['denominator'])
        tokens.append({
            'pitch': event['pitch'],
            'drum_name': event['drum_name'],
            'velocity': event['velocity'],
            'time_step': time_step,
            'measure': measure,
            'beat_in_measure': beat_in_measure
        })
    return tokens

for data in midi_data:
    data['tokens'] = tokenize_midi_events(data['drum_events'], data['bpm'], data['time_signature'])

def get_seq_length(time_signature, resolution=32):
    beats_per_measure = time_signature['numerator'] * (4 / time_signature['denominator'])
    ticks_per_measure = beats_per_measure * resolution
    return int(4 * ticks_per_measure)

dataset = []
max_seq_length = max(get_seq_length(d['time_signature'], d['bpm']) for d in midi_data)
for data in midi_data:
    seq_length = get_seq_length(data['time_signature'], data['bpm'])
    tokens = data['tokens']
    tokens.sort(key=lambda x: x['time_step'])
    if len(tokens) < seq_length:
        continue
    for i in range(0, len(tokens) - seq_length):
        seq = tokens[i:i + seq_length]
        target = tokens[i + 1:i + seq_length + 1]
        dataset.append({
            'text_input': data['text_input'],
            'midi_sequence': seq,
            'target_sequence': target
        })

train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)

def build_model(vocab_size=128, max_seq_length=max_seq_length, embedding_dim=256, num_transformer_blocks=6):
    text_input = layers.Input(shape=(4,))
    text_embedding = layers.Dense(embedding_dim)(text_input)
    text_embedding = layers.RepeatVector(max_seq_length)(text_embedding)

    midi_input = layers.Input(shape=(max_seq_length, 5))
    midi_embedding = layers.Dense(embedding_dim)(midi_input)

    x = layers.Add()([text_embedding, midi_embedding])
    for _ in range(num_transformer_blocks):
        x = layers.MultiHeadAttention(num_heads=8, key_dim=embedding_dim // 8)(x, x)
        x = layers.LayerNormalization()(x)
        x = layers.Dense(embedding_dim, activation='relu')(x)
        x = layers.LayerNormalization()(x)

    pitch_output = layers.Dense(vocab_size, name='pitch')(x)
    velocity_output = layers.Dense(128, name='velocity')(x)
    time_step_output = layers.Dense(max_seq_length, name='time_step')(x)

    model = models.Model(inputs=[text_input, midi_input], outputs=[pitch_output, velocity_output, time_step_output])
    return model

model = build_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss={
        'pitch': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        'velocity': tf.keras.losses.MeanSquaredError(),
        'time_step': tf.keras.losses.MeanSquaredError()
    },
    loss_weights={'pitch': 1.0, 'velocity': 0.5, 'time_step': 0.5}
)
model.summary()

def prepare_dataset(data):
    text_inputs = np.array([d['text_input'] for d in data])
    midi_sequences = np.array([[ [t['pitch'], t['velocity'], t['time_step'], t['measure'], t['beat_in_measure']] 
                                for t in d['midi_sequence']] for d in data])
    pitch_targets = np.array([[t['pitch'] for t in seq] for seq in [d['target_sequence'] for d in data]])
    velocity_targets = np.array([[t['velocity'] for t in seq] for seq in [d['target_sequence'] for d in data]])
    time_step_targets = np.array([[t['time_step'] for t in seq] for seq in [d['target_sequence'] for d in data]])
    return (text_inputs, midi_sequences), (pitch_targets, velocity_targets, time_step_targets)

train_inputs, train_targets = prepare_dataset(train_data)
val_inputs, val_targets = prepare_dataset(val_data)

train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_targets)).batch(6).prefetch(tf.data.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices((val_inputs, val_targets)).batch(6).prefetch(tf.data.AUTOTUNE)

model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    callbacks=[tf.keras.callbacks.ModelCheckpoint('models/music_transformer_4measures.keras', save_best_only=True)]
)

os.makedirs('models', exist_ok=True)
model.save('models/music_transformer_4measures_final.keras')
with open('models/prompt_encoder.pkl', 'wb') as f:
    pickle.dump(prompt_encoder, f)
with open('models/style_encoder.pkl', 'wb') as f:
    pickle.dump(style_encoder, f)
with open('models/time_sig_encoder.pkl', 'wb') as f:
    pickle.dump(time_sig_encoder, f)
with open('models/midi_dataset.pkl', 'wb') as f:
    pickle.dump(dataset, f)

def generate_midi(model, text_input, seed_sequence, time_signature, bpm, max_seq_length):
    seq_length = get_seq_length(time_signature)
    text_input = np.array([text_input])
    seed = np.zeros((1, max_seq_length, 5))
    for i, t in enumerate(seed_sequence[:min(len(seed_sequence), seq_length)]):
        seed[0, i] = [t['pitch'], t['velocity'], t['time_step'], t['measure'], t['beat_in_measure']]
    
    generated = []
    for _ in range(seq_length):
        pitch_pred, velocity_pred, time_step_pred = model.predict([text_input, seed], verbose=0)
        pitch = np.argmax(pitch_pred[0, -1, :])
        velocity = np.clip(np.round(velocity_pred[0, -1, 0]), 0, 127)
        time_step = np.clip(np.round(time_step_pred[0, -1, 0]), 0, seq_length - 1)
        measure = time_step // (time_signature['numerator'] * (32 // time_signature['denominator']))
        beat_in_measure = (time_step % (time_signature['numerator'] * (32 // time_signature['denominator']))) / (32 // time_signature['denominator'])
        generated.append({
            'pitch': pitch,
            'drum_name': gm_drum_map.get(pitch, f"Unknown_{pitch}"),
            'velocity': velocity,
            'time_step': time_step,
            'measure': measure,
            'beat_in_measure': beat_in_measure
        })
        seed = np.roll(seed, -1, axis=1)
        seed[0, -1] = [pitch, velocity, time_step, measure, beat_in_measure]
    
    return generated[:seq_length]

def tokens_to_midi(tokens, bpm, time_signature, output_path):
    midi = pretty_midi.PrettyMIDI()
    drum_track = pretty_midi.Instrument(program=0, is_drum=True, name='Drums')
    seconds_per_beat = 60.0 / bpm
    seconds_per_tick = seconds_per_beat / 4
    for token in tokens:
        start = token['time_step'] * seconds_per_tick
        end = start + seconds_per_tick
        note = pretty_midi.Note(velocity=int(token['velocity']), pitch=int(token['pitch']), start=start, end=end)
        drum_track.notes.append(note)
    
    midi.time_signature_changes.append(
        pretty_midi.TimeSignature(
            numerator=time_signature['numerator'],
            denominator=time_signature['denominator'],
            time=0
        )
    )
    midi.instruments.append(drum_track)
    midi.write(output_path)

def generate_from_input(model, prompt_encoder, style_encoder, time_sig_encoder, max_seq_length, dataset):
    parser = argparse.ArgumentParser(description="Generate a four-measure MIDI drum pattern.")
    parser.add_argument("--prompt", type=str, required=True, choices=list(prompt_encoder.classes_),
                        help=f"Prompt genre {list(prompt_encoder.classes_)}")
    parser.add_argument("--style", type=str, required=True, choices=list(style_encoder.classes_),
                        help=f"Style {list(style_encoder.classes_)}")
    parser.add_argument("--bpm", type=float, required=True, help="BPM (e.g., 120)")
    parser.add_argument("--time_sig", type=str, required=True,
                        help=f"Time signature {list(time_sig_encoder.classes_)}")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory for MIDI files")
    parser.add_argument("--count", type=int, default=1, help="Number of MIDI files to generate")
    parser.add_argument("--verbose", action="store_true", help="Print generated drum events")
    parser.add_argument("--seed_midi", type=str, help="Path to a MIDI file for seed sequence")

    args = parser.parse_args()

    try:
        time_sig_parts = args.time_sig.split('/')
        time_signature = {'numerator': int(time_sig_parts[0]), 'denominator': int(time_sig_parts[1])}
        valid_time_sigs = [{'numerator': 4, 'denominator': 4}, 
                          {'numerator': 3, 'denominator': 4}, 
                          {'numerator': 6, 'denominator': 8},
                          {'numerator': 5, 'denominator': 8},
                          {'numerator': 5, 'denominator': 4}]
        if time_signature not in valid_time_sigs:
            print(f"Error: Unsupported time signature {args.time_sig}. Choose from {time_sig_encoder.classes_}")
            return
    except (ValueError, IndexError):
        print("Error: Invalid time signature format. Use format like '4/4'.")
        return

    text_input = [
        args.bpm,
        style_encoder.transform([args.style])[0],
        prompt_encoder.transform([args.prompt])[0],
        time_sig_encoder.transform([args.time_sig])[0]
    ]

    if args.seed_midi:
        time_sig, drum_events, bpm, _ = parse_midi_four_measures(args.seed_midi)
        if not drum_events:
            print(f"Error: Invalid seed MIDI file {args.seed_midi}")
            return
        seed_sequence = tokenize_midi_events(drum_events, args.bpm, time_signature)
    else:
        seed_sequence = random.choice(dataset)['midi_sequence']

    os.makedirs(args.output_dir, exist_ok=True)
    for i in range(args.count):
        generated_tokens = generate_midi(model, text_input, seed_sequence, time_signature, args.bpm, max_seq_length)
        output_path = os.path.join(args.output_dir, f"generated_4measures_{args.prompt}_{args.style}_{i+1}.mid")
        tokens_to_midi(generated_tokens, args.bpm, time_signature, output_path)
        print(f"Generated MIDI saved to {output_path}")
        if args.verbose:
            for token in generated_tokens:
                print(f"Drum: {token['drum_name']}, Time: {token['time_step']}, Velocity: {token['velocity']}")

if __name__ == "__main__":
    generate_from_input(model, prompt_encoder, style_encoder, time_sig_encoder, max_seq_length, dataset)
