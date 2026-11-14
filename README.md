# Music Find

Identify a song from a short audio clip using a CNN over mel spectrograms.

## Run the web app

```bash
pip install -r requirements.txt
python backend/app.py
```

Open http://127.0.0.1:8000, then upload a clip or record one with your microphone.
(Microphone access needs `localhost` or HTTPS.)

## Layout

- `backend/app.py` – FastAPI server (`POST /api/identify`) and static hosting for `frontend/`
- `backend/inference.py` – audio → 5 s windows → mel spectrogram → `MusicCNN`, softmax averaged across windows
- `backend/labels.json` – class names in model output order (sorted folder names of `data/spectrograms/train`)
- `frontend/` – the minimalist UI; the browser converts audio to 22.05 kHz mono WAV, so ffmpeg isn't needed

If you retrain with different songs, regenerate `backend/labels.json` to match.
