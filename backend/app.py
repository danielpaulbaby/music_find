import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from inference import AudioError, Identifier

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
MAX_UPLOAD_BYTES = 30 * 1024 * 1024

state = {}


@asynccontextmanager
async def lifespan(_app):
    state["identifier"] = Identifier()
    yield


app = FastAPI(title="Music Find", lifespan=lifespan)


@app.post("/api/identify")
async def identify(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "That file is too large.")

    suffix = Path(file.filename or "").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        return state["identifier"].identify(tmp_path)
    except AudioError as exc:
        raise HTTPException(422, str(exc))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
