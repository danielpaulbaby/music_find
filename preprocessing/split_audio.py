import os
import librosa
import soundfile as sf
import numpy as np

RAW_DIR = "data/raw_songs"
CLIP_DIR = "data/clips/original"

SAMPLE_RATE = 22050
CLIP_DURATION = 5.0        # seconds
OVERLAP = 2.5              # seconds
RMS_THRESHOLD = 0.01
problem=['House of Memories']
os.makedirs(CLIP_DIR, exist_ok=True)

for song_file in os.listdir(RAW_DIR):
    if song_file in os.listdir(CLIP_DIR):
        continue
    if song_file in problem:
        continue
    if not song_file.endswith(".wav"):
        continue

    song_path = os.path.join(RAW_DIR, song_file)
    song_name = os.path.splitext(song_file)[0]

    y, sr = librosa.load(song_path, sr=SAMPLE_RATE)

    clip_samples = int(CLIP_DURATION * sr)
    hop_samples = int((CLIP_DURATION - OVERLAP) * sr)

    song_clip_dir = os.path.join(CLIP_DIR, song_name)
    os.makedirs(song_clip_dir, exist_ok=True)

    clip_count = 0

    for start in range(0, len(y) - clip_samples + 1, hop_samples):
        clip = y[start:start + clip_samples]

        rms = np.sqrt(np.mean(clip**2))
        if rms < RMS_THRESHOLD:
            continue

        clip_count += 1
        clip_path = os.path.join(
            song_clip_dir, f"clip_{clip_count:03d}.wav"
        )

        sf.write(clip_path, clip, sr)

    print(f"{song_name}: saved {clip_count} clips")
