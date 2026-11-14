import json
import re
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))
from cnn_model import MusicCNN  # noqa: E402

MODEL_PATH = ROOT / "model" / "saved_model" / "music_model_best.pth"
LABELS_PATH = Path(__file__).resolve().parent / "labels.json"

# Must match preprocessing/split_audio.py and preprocessing/generate_spectogram.py
SAMPLE_RATE = 22050
CLIP_SECONDS = 5.0
HOP_SECONDS = 2.5
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

MIN_SECONDS = 1.0
MAX_SECONDS = 60.0
SILENCE_RMS = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AudioError(ValueError):
    """The clip can't be analysed (too short, silent, unreadable)."""


_VIDEO_ID = re.compile(r"(?:\s*\[([\w-]{11})\]|[-_]([\w-]{11}))$")


def _display_name(folder_name):
    """'Bones [uZpH7EQ_PwE]' -> 'Bones', 'It_is_realme' -> 'It is realme'."""
    name = folder_name
    m = _VIDEO_ID.search(name)
    # Only treat it as a YouTube id if it mixes upper/lower case and digits/symbols,
    # so ordinary 11-letter words are left alone.
    vid = m and (m.group(1) or m.group(2))
    if vid and re.search(r"[a-z]", vid) and re.search(r"[A-Z]", vid):
        name = name[:m.start()]
    return re.sub(r"\s+", " ", name.replace("_", " ")).strip()


class Identifier:
    def __init__(self):
        self.labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
        self.model = MusicCNN(num_classes=len(self.labels)).to(DEVICE)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        self.model.eval()

    def _windows(self, y):
        clip = int(CLIP_SECONDS * SAMPLE_RATE)
        hop = int(HOP_SECONDS * SAMPLE_RATE)
        if len(y) < clip:
            y = np.tile(y, int(np.ceil(clip / len(y))))[:clip]
        starts = range(0, len(y) - clip + 1, hop)
        return [y[s:s + clip] for s in starts]

    @staticmethod
    def _spectrogram(clip):
        peak = np.max(np.abs(clip))
        if peak > 0:
            clip = clip / peak
        mel = librosa.feature.melspectrogram(
            y=clip, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
        )
        spec = librosa.power_to_db(mel, ref=np.max)
        return (spec - spec.mean()) / (spec.std() + 1e-6)

    def identify(self, path, top_k=5):
        try:
            y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=MAX_SECONDS)
        except Exception as exc:
            raise AudioError("Couldn't read that audio file.") from exc

        if len(y) < MIN_SECONDS * SAMPLE_RATE:
            raise AudioError("That clip is too short. Try at least a few seconds.")
        if np.sqrt(np.mean(y ** 2)) < SILENCE_RMS:
            raise AudioError("That clip is silent.")

        specs = [self._spectrogram(w) for w in self._windows(y)]
        batch = torch.tensor(np.stack(specs), dtype=torch.float32).unsqueeze(1).to(DEVICE)

        with torch.no_grad():
            probs = torch.softmax(self.model(batch), dim=1).mean(dim=0).cpu().numpy()

        order = np.argsort(probs)[::-1][:top_k]
        return {
            "duration": round(len(y) / SAMPLE_RATE, 1),
            "windows": len(specs),
            "matches": [
                {"title": _display_name(self.labels[i]), "confidence": float(probs[i])}
                for i in order
            ],
        }
