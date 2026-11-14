import os
import librosa
import soundfile as sf

INPUT_DIR = "data/clips"
OUTPUT_DIR = "data/clips_preprocessed"

SPLITS = ["train", "val", "test"]

SAMPLE_RATE = 22050
TRIM_SILENCE = False


# =========================
# NORMALIZATION FUNCTION
# =========================
def normalize_audio(y):
    """
    Peak normalization: scales audio between -1 and 1
    """
    max_val = max(abs(y))
    if max_val > 0:
        y = y / max_val
    return y


# =========================
# PREPROCESS SINGLE FILE
# =========================
def preprocess_file(input_path, output_path):
    try:
        # Load audio (auto converts to mono if mono=True)
        y, sr = librosa.load(input_path, sr=SAMPLE_RATE, mono=True)

        # Optional: trim silence
        if TRIM_SILENCE:
            y, _ = librosa.effects.trim(y)

        # Normalize audio
        y = normalize_audio(y)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save processed audio
        sf.write(output_path, y, SAMPLE_RATE)

        print(f"Processed: {input_path}")

    except Exception as e:
        print(f"Error processing {input_path}: {e}")


def process_split(split_name):
    """
    Processes one dataset split (train/val/test)
    """

    input_split = os.path.join(INPUT_DIR, split_name)
    output_split = os.path.join(OUTPUT_DIR, split_name)

    if not os.path.exists(input_split):
        print(f"Skipping '{split_name}' (folder not found)")
        return

    print(f"\nProcessing {split_name.upper()} set...")

    for root, _, files in os.walk(input_split):

        for file in files:

            if file.endswith(".wav"):

                input_path = os.path.join(root, file)

                relative_path = os.path.relpath(input_path, input_split)

                output_path = os.path.join(output_split, relative_path)

                preprocess_file(input_path, output_path)


if __name__ == "__main__":

    for split in SPLITS:
        process_split(split)

    print("\nAudio preprocessing completed successfully.")