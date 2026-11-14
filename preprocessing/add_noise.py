import os
import random
import librosa
import numpy as np
import soundfile as sf

# =========================
# CONFIG
# =========================
INPUT_DIR = "data/clips_preprocessed/train"
NOISE_DIR = "data/noise_samples"
OUTPUT_DIR = "data/clips_noisy/train"

SAMPLE_RATE = 22050

# SNR range in dB
SNR_MIN = 5
SNR_MAX = 20


# =========================
# LOAD RANDOM NOISE
# =========================
def load_random_noise():
    noise_files = []

    # Walk through all subdirectories
    for root, _, files in os.walk(NOISE_DIR):
        for file in files:
            if file.endswith((".wav", ".mp3")):
                noise_files.append(os.path.join(root, file))

    if len(noise_files) == 0:
        raise ValueError("No noise files found in noise_samples (including subfolders).")

    noise_path = random.choice(noise_files)
    noise, _ = librosa.load(noise_path, sr=SAMPLE_RATE)

    return noise


# =========================
# MATCH NOISE LENGTH
# =========================
def match_length(noise, target_len):
    if len(noise) > target_len:
        noise = noise[:target_len]
    else:
        repeat = int(np.ceil(target_len / len(noise)))
        noise = np.tile(noise, repeat)[:target_len]
    return noise


# =========================
# ADD NOISE WITH SNR
# =========================
def add_noise_to_signal(signal, noise):
    # Match lengths
    noise = match_length(noise, len(signal))

    # Compute signal power
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return signal

    # Random SNR
    snr_db = random.uniform(SNR_MIN, SNR_MAX)

    # Scale noise
    snr_linear = 10 ** (snr_db / 10)
    scale = np.sqrt(signal_power / (snr_linear * noise_power))

    noisy_signal = signal + scale * noise

    # Normalize to avoid clipping
    max_val = np.max(np.abs(noisy_signal))
    if max_val > 1:
        noisy_signal = noisy_signal / max_val

    return noisy_signal


# =========================
# PROCESS SINGLE FILE
# =========================
def process_file(input_path, output_path):
    try:
        signal, _ = librosa.load(input_path, sr=SAMPLE_RATE)

        noise = load_random_noise()
        noisy_signal = add_noise_to_signal(signal, noise)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, noisy_signal, SAMPLE_RATE)

        print(f"Noisy: {input_path}")

    except Exception as e:
        print(f"Error: {input_path} -> {e}")


# =========================
# PROCESS ALL FILES
# =========================
def process_all():
    for root, _, files in os.walk(INPUT_DIR):
        for file in files:
            if file.endswith(".wav"):
                input_path = os.path.join(root, file)

                relative_path = os.path.relpath(input_path, INPUT_DIR)
                output_path = os.path.join(OUTPUT_DIR, relative_path)

                # rename file
                output_path = output_path.replace(".wav", "_noisy.wav")

                process_file(input_path, output_path)


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    process_all()