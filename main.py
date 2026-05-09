import base64
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse


@dataclass
class AnalyzerConfig:
    min_visibility: float = 0.4
    hip_tilt_threshold_deg: float = 14.0
    knee_min_deg: float = 120.0
    knee_max_deg: float = 176.0
    alarm_speed_threshold_px_s: float = 280.0
    fast_motion_threshold_px: float = 12.0


def point_speed_px_per_sec(
    prev_point: Optional[Dict[str, float]], current_point: Optional[Dict[str, float]], dt_sec: float
) -> float:
    if not prev_point or not current_point:
        return 0.0
    dx = current_point["x"] - prev_point["x"]
    dy = current_point["y"] - prev_point["y"]
    return float(np.sqrt(dx * dx + dy * dy) / max(dt_sec, 1e-3))


def detect_fast_motion(
    prev_keypoints: Optional[Dict[str, Dict[str, float]]],
    current_keypoints: Dict[str, Dict[str, float]],
    threshold_px: float,
) -> bool:
    if not prev_keypoints:
        return False
    for joint in ("LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"):
        p = prev_keypoints.get(joint)
        c = current_keypoints.get(joint)
        if not p or not c:
            continue
        if np.hypot(c["x"] - p["x"], c["y"] - p["y"]) > threshold_px:
            return True
    return False


def smooth_keypoints(
    prev_keypoints: Optional[Dict[str, Dict[str, float]]],
    current_keypoints: Dict[str, Dict[str, float]],
    fast_motion: bool = False,
) -> Dict[str, Dict[str, float]]:
    if not prev_keypoints:
        return current_keypoints

    # Default EMA: 0.7/0.3. During fast kicks, lower alpha to reduce lag.
    if fast_motion:
        alpha_prev = 0.35
        alpha_curr = 0.65
    else:
        # Slightly snappier than 0.7/0.3 to reduce visible skeleton lag vs the video.
        alpha_prev = 0.6
        alpha_curr = 0.4

    smoothed: Dict[str, Dict[str, float]] = {}
    for name, current in current_keypoints.items():
        prev = prev_keypoints.get(name)
        if not prev:
            smoothed[name] = current
            continue
        smoothed[name] = {
            "x": float(prev["x"] * alpha_prev + current["x"] * alpha_curr),
            "y": float(prev["y"] * alpha_prev + current["y"] * alpha_curr),
            "z": float(prev["z"] * alpha_prev + current["z"] * alpha_curr),
            "visibility": float(prev["visibility"] * alpha_prev + current["visibility"] * alpha_curr),
        }
    return smoothed


class FootballAnalyzer:
    def __init__(self, config: Optional[AnalyzerConfig] = None) -> None:
        self.config = config or AnalyzerConfig()
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    @staticmethod
    def _decode_frame(data_url: str) -> Optional[np.ndarray]:
        try:
            encoded = data_url.split(",", 1)[1] if "," in data_url else data_url
            raw = base64.b64decode(encoded)
            arr = np.frombuffer(raw, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return None

    @staticmethod
    def _line_angle_deg(a: Dict[str, float], b: Dict[str, float]) -> float:
        dy = b["y"] - a["y"]
        dx = b["x"] - a["x"]
        return float(np.degrees(np.arctan2(dy, dx)))

    @staticmethod
    def _joint_angle_deg(a: Dict[str, float], b: Dict[str, float], c: Dict[str, float]) -> float:
        ba = np.array([a["x"] - b["x"], a["y"] - b["y"]], dtype=np.float64)
        bc = np.array([c["x"] - b["x"], c["y"] - b["y"]], dtype=np.float64)
        if np.linalg.norm(ba) < 1e-6 or np.linalg.norm(bc) < 1e-6:
            return 180.0
        cosine = np.clip(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)), -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine)))

    def _extract_all_keypoints(self, landmarks, width: int, height: int) -> Dict[str, Dict[str, float]]:
        points: Dict[str, Dict[str, float]] = {}
        for lm_enum in self.mp_pose.PoseLandmark:
            lm = landmarks.landmark[lm_enum.value]
            points[lm_enum.name] = {
                "x": float(lm.x * width),
                "y": float(lm.y * height),
                "z": float(lm.z),
                "visibility": float(lm.visibility),
            }
        return points

    def _required_visible(self, keypoints: Dict[str, Dict[str, float]]) -> bool:
        required = ("LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE")
        for name in required:
            lm = keypoints.get(name)
            if not lm or lm["visibility"] < self.config.min_visibility:
                return False
        return True

    def evaluate(
        self,
        frame_b64: str,
        prev_keypoints: Optional[Dict[str, Dict[str, float]]] = None,
        prev_ts_ms: Optional[int] = None,
    ) -> Dict:
        start = time.perf_counter()
        now_ms = int(time.time() * 1000)
        frame = self._decode_frame(frame_b64)
        if frame is None:
            return {
                "status": "NO_POSE",
                "reason": "invalid_frame",
                "keypoints": {},
                "kick_speed": 0.0,
                "knee_angle": 180.0,
                "posture_consistency_score": 0.0,
                "metrics": {},
                "inference_time_ms": round((time.perf_counter() - start) * 1000, 2),
                "timestamp_ms": now_ms,
            }

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)
        if not result.pose_landmarks:
            return {
                "status": "NO_POSE",
                "reason": "no_pose",
                "keypoints": {},
                "kick_speed": 0.0,
                "knee_angle": 180.0,
                "posture_consistency_score": 0.0,
                "metrics": {},
                "inference_time_ms": round((time.perf_counter() - start) * 1000, 2),
                "timestamp_ms": now_ms,
            }

        raw_keypoints = self._extract_all_keypoints(result.pose_landmarks, w, h)
        fast_motion = detect_fast_motion(prev_keypoints, raw_keypoints, self.config.fast_motion_threshold_px)
        keypoints = smooth_keypoints(prev_keypoints, raw_keypoints, fast_motion)
        if not self._required_visible(keypoints):
            return {
                "status": "NO_POSE",
                "reason": "low_visibility",
                "keypoints": keypoints,
                "kick_speed": 0.0,
                "knee_angle": 180.0,
                "posture_consistency_score": 0.0,
                "metrics": {},
                "smoothed_keypoints": keypoints,
                "inference_time_ms": round((time.perf_counter() - start) * 1000, 2),
                "timestamp_ms": now_ms,
            }

        dt_sec = ((now_ms - prev_ts_ms) / 1000.0) if prev_ts_ms else (1.0 / 30.0)
        dt_sec = max(1e-3, dt_sec)
        left_speed = point_speed_px_per_sec(
            prev_keypoints.get("LEFT_ANKLE") if prev_keypoints else None, keypoints.get("LEFT_ANKLE"), dt_sec
        )
        right_speed = point_speed_px_per_sec(
            prev_keypoints.get("RIGHT_ANKLE") if prev_keypoints else None, keypoints.get("RIGHT_ANKLE"), dt_sec
        )
        kicking_leg = "LEFT" if left_speed >= right_speed else "RIGHT"
        kick_speed = left_speed if kicking_leg == "LEFT" else right_speed

        if kicking_leg == "LEFT":
            hip, knee, ankle = keypoints.get("LEFT_HIP"), keypoints.get("LEFT_KNEE"), keypoints.get("LEFT_ANKLE")
        else:
            hip, knee, ankle = keypoints.get("RIGHT_HIP"), keypoints.get("RIGHT_KNEE"), keypoints.get("RIGHT_ANKLE")
        knee_angle = self._joint_angle_deg(hip, knee, ankle) if hip and knee and ankle else 180.0

        hip_tilt_deg = abs(self._line_angle_deg(keypoints["LEFT_HIP"], keypoints["RIGHT_HIP"]))
        if hip_tilt_deg > 90.0:
            hip_tilt_deg = abs(180.0 - hip_tilt_deg)

        unstable_hip = hip_tilt_deg > self.config.hip_tilt_threshold_deg
        bad_knee = knee_angle < self.config.knee_min_deg or knee_angle > self.config.knee_max_deg
        is_alarm = kick_speed > self.config.alarm_speed_threshold_px_s and (unstable_hip or bad_knee)
        status = "ALARM" if is_alarm else "OK"

        velocity_score = max(0.0, min(100.0, (kick_speed / 900.0) * 100.0))
        if status == "ALARM":
            velocity_score = min(velocity_score, 55.0)

        return {
            "status": status,
            "reason": "strike_risk" if is_alarm else "strike_ok",
            "keypoints": keypoints,
            "kick_speed": round(float(kick_speed), 2),
            "knee_angle": round(float(knee_angle), 2),
            "kicking_leg": kicking_leg,
            "posture_consistency_score": round(float(velocity_score), 1),
            "metrics": {
                "hip_tilt_deg": round(float(hip_tilt_deg), 2),
                "left_ankle_speed_px_s": round(float(left_speed), 2),
                "right_ankle_speed_px_s": round(float(right_speed), 2),
                "fast_motion_mode": fast_motion,
            },
            # Includes decode + model + EMA + strike analytics.
            "inference_time_ms": round((time.perf_counter() - start) * 1000, 2),
            "smoothed_keypoints": keypoints,
            "timestamp_ms": now_ms,
        }


app = FastAPI(title="NeuroStrike Football Action Analyzer")
analyzer: Optional[FootballAnalyzer] = None
uploaded_videos: Dict[str, str] = {}
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


def get_analyzer() -> Optional[FootballAnalyzer]:
    global analyzer
    if analyzer is None:
        try:
            analyzer = FootballAnalyzer()
        except Exception:
            analyzer = None
    return analyzer


@dataclass
class SessionState:
    prev_keypoints: Dict[str, Dict[str, float]]
    prev_ts_ms: Optional[int]
    mode: str


@app.get("/")
async def root() -> HTMLResponse:
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)) -> Dict[str, str]:
    if not file.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only .mp4 files are supported")
    file_id = uuid.uuid4().hex
    target = UPLOAD_DIR / f"{file_id}.mp4"
    target.write_bytes(await file.read())
    uploaded_videos[file_id] = str(target)
    return {"video_file_id": file_id}


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    session = SessionState(prev_keypoints={}, prev_ts_ms=None, mode="WEBCAM")
    try:
        while True:
            payload = await websocket.receive_json()
            client_ts = payload.get("client_ts")
            frame_seq_in = payload.get("frame_seq")
            mode = str(payload.get("mode", session.mode)).upper()
            session.mode = mode

            model = get_analyzer()
            if model is None:
                await websocket.send_json(
                    {
                        "status": "NO_POSE",
                        "reason": "mediapipe_init_failed",
                        "keypoints": {},
                        "kick_speed": 0.0,
                        "knee_angle": 180.0,
                        "posture_consistency_score": 0.0,
                        "metrics": {},
                        "inference_time_ms": 0.0,
                        "client_ts": client_ts,
                        "frame_seq": frame_seq_in,
                        "server_ts": int(time.time() * 1000),
                        "source_mode": mode,
                    }
                )
                continue

            # VIDEO_FILE must send frames from the client (same as WEBCAM) so skeleton
            # stays locked to the frame the user sees. Server-side full decode was async
            # with browser playback and caused large spatial/temporal drift.

            frame_data = payload.get("frame")
            if not frame_data:
                await websocket.send_json(
                    {
                        "status": "NO_POSE",
                        "reason": "missing_frame",
                        "keypoints": {},
                        "kick_speed": 0.0,
                        "knee_angle": 180.0,
                        "posture_consistency_score": 0.0,
                        "metrics": {},
                        "inference_time_ms": 0.0,
                        "client_ts": client_ts,
                        "frame_seq": frame_seq_in,
                        "server_ts": int(time.time() * 1000),
                        "source_mode": mode,
                    }
                )
                continue

            frame_seq = frame_seq_in
            result = model.evaluate(frame_data, session.prev_keypoints, session.prev_ts_ms)
            session.prev_keypoints = result.get("smoothed_keypoints", {})
            session.prev_ts_ms = result.get("timestamp_ms")
            result.pop("smoothed_keypoints", None)
            result["client_ts"] = client_ts
            result["frame_seq"] = frame_seq
            result["server_ts"] = int(time.time() * 1000)
            result["source_mode"] = mode
            await websocket.send_json(result)
    except WebSocketDisconnect:
        return

