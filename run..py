def generate_from_input(model, prompt_encoder, style_encoder, time_sig_encoder, max_seq_length, dataset):
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate a four-measure MIDI drum pattern.")
    parser.add_argument("--prompt", type=str, required=True, choices=['funk', 'rock', 'hiphop'],
                        help="Prompt genre (funk, rock, hiphop)")
    parser.add_argument("--style", type=str, required=True, choices=['beat', 'fill'],
                        help="Style (beat, fill)")
    parser.add_argument("--bpm", type=float, required=True,
                        help="BPM (e.g., 120)")
    parser.add_argument("--time_sig", type=str, required=True,
                        help="Time signature (e.g., 4/4, 3/4, 6/8, 5/8)")

    # Parse arguments
    args = parser.parse_args()

    # Validate time signature
    try:
        time_sig_parts = args.time_sig.split('/')
        time_signature = {'numerator': int(time_sig_parts[0]), 'denominator': int(time_sig_parts[1])}
        valid_time_sigs = [{'numerator': 4, 'denominator': 4}, 
                          {'numerator': 3, 'denominator': 4}, 
                          {'numerator': 6, 'denominator': 8},
                          {'numerator': 5, 'denominator': 8}]
        if time_signature not in valid_time_sigs:
            print(f"Error: Unsupported time signature {args.time_sig}. Choose from 4/4, 3/4, 6/8, 5/8.")
            return
    except (ValueError, IndexError):
        print("Error: Invalid time signature format. Use format like '4/4'.")
        return

    # Validate encoders have the required classes
    if args.prompt not in prompt_encoder.classes_:
        print(f"Error: Prompt '{args.prompt}' not in trained prompts: {prompt_encoder.classes_}")
        return
    if args.style not in style_encoder.classes_:
        print(f"Error: Style '{args.style}' not in trained styles: {style_encoder.classes_}")
        return
    if args.time_sig not in time_sig_encoder.classes_:
        print(f"Error: Time signature '{args.time_sig}' not in trained time signatures: {time_sig_encoder.classes_}")
        return

    # Create text input
    text_input = [
        args.bpm,
        style_encoder.transform([args.style])[0],
        prompt_encoder.transform([args.prompt])[0],
        time_sig_encoder.transform([args.time_sig])[0]
    ]

    # Use a seed sequence from dataset (or load from a MIDI file if needed)
    seed_sequence = dataset[0]['midi_sequence']  # First sequence from dataset

    # Generate MIDI
    generated_tokens = generate_midi(model, text_input, seed_sequence, time_signature, args.bpm, max_seq_length)

    # Save output
    output_path = f"generated_4measures_{args.prompt}_{args.style}.mid"
    tokens_to_midi(generated_tokens, args.bpm, time_signature, output_path)
    print(f"Generated MIDI saved to {output_path}")

if __name__ == "__main__":
    # Load model and tokenizers
    model = tf.keras.models.load_model('music_transformer_4measures_final.keras')
    with open('prompt_encoder.pkl', 'rb') as f: