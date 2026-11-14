import os
import librosa
import numpy as np

# =========================
# CONFIG
# =========================

# Clean audio
CLEAN_TRAIN_DIR = "data/clips_preprocessed/train"
CLEAN_VAL_DIR = "data/clips_preprocessed/val"
CLEAN_TEST_DIR = "data/clips_preprocessed/test"

# Noisy audio (training only)
NOISY_TRAIN_DIR = "data/clips_noisy/train"

# Output directory
OUTPUT_DIR = "data/spectrograms"

# Spectrogram parameters
SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512


# =========================
# GENERATE MEL SPECTROGRAM
# =========================
def generate_mel_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    return log_mel_spec


# =========================
# PROCESS DIRECTORY
# =========================
def process_directory(input_dir, output_subdir):

    if not os.path.exists(input_dir):
        print(f"Skipping: {input_dir} (Folder not found)")
        return

    for root, _, files in os.walk(input_dir):

        for file in files:

            if file.endswith(".wav"):

                input_path = os.path.join(root, file)

                # Generate spectrogram
                spec = generate_mel_spectrogram(input_path)

                # Preserve folder structure
                relative_path = os.path.relpath(input_path, input_dir)

                output_path = os.path.join(
                    OUTPUT_DIR,
                    output_subdir,
                    relative_path.replace(".wav", ".npy")
                )

                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                np.save(output_path, spec)

                print(f"Saved: {output_path}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    print("\nGenerating TRAIN spectrograms (Clean)...")
    process_directory(CLEAN_TRAIN_DIR, "train")

    print("\nGenerating TRAIN spectrograms (Noisy)...")
    process_directory(NOISY_TRAIN_DIR, "train")

    print("\nGenerating VALIDATION spectrograms...")
    process_directory(CLEAN_VAL_DIR, "val")

    print("\nGenerating TEST spectrograms...")
    process_directory(CLEAN_TEST_DIR, "test")

    print("\nSpectrogram generation completed successfully.")