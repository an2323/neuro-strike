"""
NeuroStrike Strike Lab — Flask API for offline ghost-overlay video analysis.

Run from repo root:
  pip install -r requirements.txt
  bash scripts/fix_venv_opencv_numpy.sh   # once, if cv2/numpy issues
  python app.py

By default the app is served with **Waitress** (no Werkzeug dev warning).
For Flask’s built-in dev server only:  STRIKE_LAB_DEV=1 python app.py
Port / host: STRIKE_LAB_PORT (default 5050), STRIKE_LAB_HOST (default 0.0.0.0).

Routes:
  GET  /                 — Analyzer UI (dark / neon)
  POST /analyze-video    — Upload .mp4 / .mov, run strike_video_processor.process_video
  GET  /results/<name>   — Stream processed MP4
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename

# strike_video_processor pulls cv2 / mediapipe — import lazily so a broken numpy/opencv
# venv still serves GET / and static UI until deps are repaired (see requirements.txt).
def _run_process_video(
    input_path: Path,
    output_path: Path,
    min_det: float = 0.5,
    min_trk: float = 0.5,
) -> dict:
    from strike_video_processor import process_video

    return process_video(input_path, output_path, min_det, min_trk)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("strike_lab")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
ALLOWED_UPLOAD_EXT = {".mp4", ".mov"}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"), static_folder=str(BASE_DIR / "static"))
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024  # 512 MB


@app.route("/")
def index():
    return render_template("strike_index.html")


@app.route("/analyze-video", methods=["POST"])
def analyze_video():
    if "video" not in request.files:
        logger.warning("analyze-video: missing 'video' file field")
        return jsonify({"ok": False, "error": "Missing file field 'video' (multipart)."}), 400

    f = request.files["video"]
    if not f or f.filename == "":
        return jsonify({"ok": False, "error": "No file selected."}), 400

    raw_name = secure_filename(f.filename)
    ext = Path(raw_name).suffix.lower()
    if ext not in ALLOWED_UPLOAD_EXT:
        return jsonify({"ok": False, "error": f"Only {sorted(ALLOWED_UPLOAD_EXT)} are allowed."}), 400

    ts = int(time.time() * 1000)
    uid = uuid.uuid4().hex[:10]
    upload_name = f"upload_{ts}_{uid}{ext}"
    result_name = f"result_{ts}_{uid}.mp4"
    report_name = f"result_{ts}_{uid}_storyboard.png"
    upload_path = UPLOAD_DIR / upload_name
    result_path = RESULTS_DIR / result_name
    report_path = RESULTS_DIR / report_name

    try:
        f.save(str(upload_path))
        logger.info("Saved upload %s (%s bytes)", upload_name, upload_path.stat().st_size)
    except OSError as e:
        logger.exception("Failed to save upload")
        return jsonify({"ok": False, "error": f"Could not save upload: {e}"}), 500

    t0 = time.time()
    try:
        result_meta = _run_process_video(upload_path, result_path)
    except ImportError as e:
        logger.exception("Strike processor import failed (numpy/opencv/mediapipe stack)")
        try:
            upload_path.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            result_path.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            report_path.unlink(missing_ok=True)
        except OSError:
            pass
        return jsonify(
            {
                "ok": False,
                "error": (
                    "Video stack failed to load (numpy/cv2; often opencv-contrib-python vs headless). "
                    "On the server with venv active:  bash scripts/fix_venv_opencv_numpy.sh — "
                    f"Details: {e}"
                ),
            }
        ), 500
    except RuntimeError as e:
        logger.error("Processor failed (RuntimeError): %s", e)
        try:
            upload_path.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            result_path.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            report_path.unlink(missing_ok=True)
        except OSError:
            pass
        return jsonify({"ok": False, "error": str(e)}), 500
    except Exception:
        logger.exception("Processor failed (unexpected)")
        try:
            upload_path.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            result_path.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            report_path.unlink(missing_ok=True)
        except OSError:
            pass
        return jsonify({"ok": False, "error": "Video processing failed. See server logs."}), 500

    video_name = Path(result_meta.get("video_path", str(result_path))).name
    report_name = Path(
        result_meta.get("report_path", str(result_path.with_name(result_path.stem + "_storyboard.png")))
    ).name
    analysis_time_sec = round(max(0.0, time.time() - t0), 2)
    video_url = url_for("serve_result", filename=video_name)
    report_url = url_for("serve_result", filename=report_name)
    coaching_data = result_meta.get("coaching_data", {})
    coaching_data["analysis_time_sec"] = analysis_time_sec
    orbit_3d_url = None
    orbit_meta = result_meta.get("orbit_3d_path")
    if orbit_meta:
        orbit_name = Path(str(orbit_meta)).name
        orbit_path = RESULTS_DIR / orbit_name
        if orbit_path.is_file():
            orbit_3d_url = url_for("serve_result", filename=orbit_name)
    payload = {
        "ok": True,
        "video_filename": video_name,
        "report_filename": report_name,
        "video_url": video_url,
        "report_url": report_url,
        "orbit_3d_url": orbit_3d_url,
        "coaching_data": coaching_data,
    }
    logger.info("Analysis complete -> video=%s report=%s time=%.2fs", video_name, report_name, analysis_time_sec)
    return jsonify(payload)


@app.route("/results/<filename>")
def serve_result(filename: str):
    if "/" in filename or "\\" in filename or filename.startswith("."):
        abort(400)
    safe = secure_filename(filename)
    if safe != filename:
        abort(400)
    path = RESULTS_DIR / safe
    if not path.is_file():
        abort(404)
    ext = path.suffix.lower()
    if ext == ".png":
        mimetype = "image/png"
    elif ext == ".mp4":
        mimetype = "video/mp4"
    else:
        mimetype = None
    return send_file(path, mimetype=mimetype, as_attachment=False, download_name=safe, conditional=True)


if __name__ == "__main__":
    host = os.environ.get("STRIKE_LAB_HOST", "0.0.0.0")
    port = int(os.environ.get("STRIKE_LAB_PORT", "5050"))
    if os.environ.get("STRIKE_LAB_DEV", "").lower() in ("1", "true", "yes"):
        logger.warning("STRIKE_LAB_DEV: using Flask development server")
        app.run(host=host, port=port, threaded=True)
    else:
        try:
            from waitress import serve

            logger.info("Strike Lab: Waitress on http://%s:%s/", host, port)
            serve(app, host=host, port=port, threads=4)
        except ImportError:
            logger.warning("waitress missing; pip install waitress — using Flask dev server")
            app.run(host=host, port=port, threaded=True)
