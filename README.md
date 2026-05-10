# NeuroStrike

**NeuroStrike** is a football strike biomechanics stack: pose estimation on strike footage, ghost / heatmap-style overlays, coaching metrics, and an optional **narrated cinematic analysis** export. It is built for **AMD ROCm** and **Instinct MI300-class** accelerators (MediaPipe GPU enabled by default on the remote path).

The repo contains **three runnable surfaces** that share the same biomechanical ideas but target different workflows:

| Component | Role |
|-----------|------|
| **Strike Lab** (`app.py`) | Web UI + **Flask** API: upload `.mp4` / `.mov`, run offline processing, download narrated MP4, storyboard PNG, and structured coaching JSON. |
| **`strike_video_processor.py`** | CLI / library: full pipeline (MediaPipe → smoothing → narrated video + storyboard). Used by Strike Lab. |
| **`remote_main.py`** | **FastAPI** + **WebSockets**: high-concurrency remote analyzer for MI300X; binary + JPEG streaming (see code for protocol). |
| **`main.py`** | Smaller **FastAPI** app (e.g. local/dev) with upload/WebSocket paths; may default MediaPipe to CPU — check env in file. |

---

## What Strike Lab does

1. **Ingest** a short strike clip (MP4 or MOV).
2. **Estimate pose** with **MediaPipe Pose** at **model complexity 2** (33 landmarks).
3. **Smooth** trajectories (Savitzky–Golay where SciPy is available).
4. **Detect** strike phase, kicking leg, and biomechanical signals (e.g. ankle speed, joint angles, form match vs a corrected “ghost” template).
5. **Produce**:
   - A single **narrated analysis MP4** (overlay + freeze segments + **gTTS** commentary, assembled with **MoviePy**).
   - A **three-panel storyboard** PNG.
   - **Coaching payload** (scores, strengths/weaknesses, drill suggestions) returned as JSON to the browser.

Processing expects **Linux + Python 3.10+** and a sane **ffmpeg/ffprobe** on `PATH` for H.264 remux and audio handling.

---

## Main technologies

- **Python 3** — NumPy, SciPy (temporal smoothing).
- **OpenCV** (`opencv-python-headless`) — decode/encode, overlays.
- **MediaPipe** — BlazePose / Pose, GPU-friendly on ROCm when `MEDIAPIPE_DISABLE_GPU=0`.
- **Flask** + **Waitress** — Strike Lab HTTP server.
- **FastAPI** + **Uvicorn** — `remote_main.py` / `main.py`.
- **MoviePy** + **gTTS** — narrated export (optional deps are listed in `requirements.txt` comments).
- **ffmpeg** — browser-friendly H.264 + AAC pass after export.

Frontend for Strike Lab: **vanilla HTML/CSS/JS** (`templates/strike_index.html`, `static/script.js`).

---

## Quick start (Strike Lab)

From the repository root:

```bash
python3 -m venv neurostrike_env
source neurostrike_env/bin/activate   # Windows: neurostrike_env\Scripts\activate
pip install -r requirements.txt
bash scripts/fix_venv_opencv_numpy.sh   # strongly recommended once (OpenCV / NumPy / MediaPipe conflicts)
python app.py
```

Then open **http://127.0.0.1:5050/** (or host/port from env: `STRIKE_LAB_HOST`, `STRIKE_LAB_PORT`).

- **`STRIKE_LAB_DEV=1 python app.py`** — Flask development server instead of Waitress.

### Optional: remote MI300X stack

```bash
pip install -r requirements_remote.txt
bash scripts/fix_venv_opencv_numpy.sh   # if needed
# run per remote_main.py / deployment docs
```

---

## CLI (offline processor)

```bash
python strike_video_processor.py --input path/to/kick.mp4 --output path/to/out.mp4
```

Same pipeline as Strike Lab’s backend (narrated output path, storyboard, coaching). Additional optional deps may be listed in `requirements_strike_video.txt`.

---

## Project layout (high level)

```
app.py                    # Strike Lab Flask entry
templates/               # Strike Lab HTML
static/                    # Strike Lab JS/CSS
strike_video_processor.py # Core offline biomechanics + narrated export
remote_main.py            # FastAPI WebSocket server (MI300X / cloud)
main.py                   # Alternate FastAPI entry
scripts/                   # e.g. OpenCV/NumPy venv fix
uploads/  results/        # Runtime dirs for Strike Lab (created by app)
```

---

## Hardware notes

- **ROCm / MI300X**: Keep GPU enabled for MediaPipe on the remote server (`MEDIAPIPE_DISABLE_GPU=0` is the default in `remote_main.py`; align Strike Lab / workers with your deployment policy).
- Strike Lab’s **upload processor** is CPU-heavy (OpenCV + MediaPipe); GPU use depends on MediaPipe build and env — see MediaPipe and ROCm docs for your image.

---

## License / product

Add your license and trademark text here if applicable.
