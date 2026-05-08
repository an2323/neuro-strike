import base64
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional

os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse


@dataclass
class PostureConfig:
    min_visibility: float = 0.4
    y_delta_threshold: float = 0.1
    shoulder_tilt_threshold_deg: float = 12.0
    baseline_drop_ratio: float = 0.88
    baseline_release_ratio: float = 0.94
    alarm_frames_to_trigger: int = 3
    ok_frames_to_release: int = 2


class PostureAnalyzer:
    """MediaPipe-based posture analyzer.

    Encapsulates posture logic so a heavier model can replace MediaPipe later
    without changing the WebSocket transport contract.
    """

    def __init__(self, config: Optional[PostureConfig] = None) -> None:
        self.config = config or PostureConfig()
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
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
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return frame
        except Exception:
            return None

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

    def _validate_required_keypoints(
        self, keypoints: Dict[str, Dict[str, float]]
    ) -> Optional[Dict[str, Dict[str, float]]]:
        required = {
            "LEFT_EAR": self.mp_pose.PoseLandmark.LEFT_EAR,
            "RIGHT_EAR": self.mp_pose.PoseLandmark.RIGHT_EAR,
            "LEFT_SHOULDER": self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            "RIGHT_SHOULDER": self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
        }
        for name, idx in required.items():
            lm = keypoints.get(idx.name)
            if not lm or lm["visibility"] < self.config.min_visibility:
                return None
        return keypoints

    @staticmethod
    def _line_angle_deg(a: Dict[str, float], b: Dict[str, float]) -> float:
        dy = b["y"] - a["y"]
        dx = b["x"] - a["x"]
        return float(np.degrees(np.arctan2(dy, dx)))

    def evaluate(
        self, frame_b64: str, prev_keypoints: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Dict:
        start = time.perf_counter()
        frame = self._decode_frame(frame_b64)
        if frame is None:
            return {
                "status": "NO_POSE",
                "reason": "invalid_frame",
                "keypoints": {},
                "metrics": {},
                "inference_time_ms": round((time.perf_counter() - start) * 1000, 2),
            }

        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)
        if not result.pose_landmarks:
            return {
                "status": "NO_POSE",
                "reason": "no_pose",
                "keypoints": {},
                "metrics": {},
                "inference_time_ms": round((time.perf_counter() - start) * 1000, 2),
            }

        current_keypoints = self._extract_all_keypoints(result.pose_landmarks, width, height)
        keypoints = smooth_keypoints(prev_keypoints, current_keypoints)
        if self._validate_required_keypoints(keypoints) is None:
            return {
                "status": "NO_POSE",
                "reason": "low_visibility",
                "keypoints": {},
                "metrics": {},
                "smoothed_keypoints": keypoints,
                "inference_time_ms": round((time.perf_counter() - start) * 1000, 2),
            }

        left_delta_y = keypoints["LEFT_SHOULDER"]["y"] - keypoints["LEFT_EAR"]["y"]
        right_delta_y = keypoints["RIGHT_SHOULDER"]["y"] - keypoints["RIGHT_EAR"]["y"]
        avg_ear_shoulder_delta = (left_delta_y + right_delta_y) / 2.0
        raw_shoulder_angle = self._line_angle_deg(
            keypoints["LEFT_SHOULDER"], keypoints["RIGHT_SHOULDER"]
        )
        # Convert to "tilt away from horizontal" in [0, 90].
        shoulder_tilt_deg = abs(raw_shoulder_angle)
        if shoulder_tilt_deg > 90.0:
            shoulder_tilt_deg = abs(180.0 - shoulder_tilt_deg)

        # Slump if ear-to-shoulder vertical distance gets too small
        # OR if shoulder alignment is heavily tilted.
        is_slump = (
            avg_ear_shoulder_delta < (height * self.config.y_delta_threshold)
            or shoulder_tilt_deg > self.config.shoulder_tilt_threshold_deg
        )
        status = "ALARM" if is_slump else "OK"

        return {
            "status": status,
            "reason": "slump_detected" if is_slump else "posture_ok",
            "keypoints": keypoints,
            "posture_consistency_score": 0.0,
            "metrics": {
                "avg_ear_shoulder_delta_px": round(float(avg_ear_shoulder_delta), 2),
                "left_ear_shoulder_delta_px": round(float(left_delta_y), 2),
                "right_ear_shoulder_delta_px": round(float(right_delta_y), 2),
                "shoulder_tilt_deg": round(float(shoulder_tilt_deg), 2),
            },
            "inference_time_ms": round((time.perf_counter() - start) * 1000, 2),
            "smoothed_keypoints": keypoints,
        }


app = FastAPI(title="NeuroStrike Local MVP")
analyzer: Optional[PostureAnalyzer] = None


def get_analyzer() -> Optional[PostureAnalyzer]:
    global analyzer
    if analyzer is None:
        try:
            analyzer = PostureAnalyzer()
        except Exception:
            analyzer = None
    return analyzer


@dataclass
class SessionState:
    baseline_samples: list[float]
    baseline_delta_px: Optional[float]
    alarm_streak: int
    ok_streak: int
    stable_status: str
    prev_keypoints: Dict[str, Dict[str, float]]


def smooth_keypoints(
    prev_keypoints: Optional[Dict[str, Dict[str, float]]], current_keypoints: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    if not prev_keypoints:
        return current_keypoints

    alpha_prev = 0.7
    alpha_curr = 0.3
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


def apply_session_calibration(result: Dict, session: SessionState) -> Dict:
    if result.get("status") == "NO_POSE":
        result["posture_consistency_score"] = 0.0
        return result

    metrics = result.get("metrics", {})
    avg_delta = metrics.get("avg_ear_shoulder_delta_px")
    if avg_delta is None:
        return result

    if session.baseline_delta_px is None:
        session.baseline_samples.append(float(avg_delta))
        if len(session.baseline_samples) >= 24:
            session.baseline_delta_px = float(np.median(session.baseline_samples))
        result["status"] = "CALIBRATING"
        result["reason"] = "collecting_baseline"
        progress = min(100.0, (len(session.baseline_samples) / 24.0) * 100.0)
        result["posture_consistency_score"] = round(progress, 1)
        result["metrics"]["baseline_delta_px"] = (
            round(session.baseline_delta_px, 2) if session.baseline_delta_px else None
        )
        return result

    config = PostureConfig()
    baseline = session.baseline_delta_px
    drop_threshold = baseline * config.baseline_drop_ratio
    release_threshold = baseline * config.baseline_release_ratio
    tilt = float(metrics.get("shoulder_tilt_deg", 0.0))
    strong_alarm = float(avg_delta) < drop_threshold or tilt > config.shoulder_tilt_threshold_deg
    clear_ok = float(avg_delta) > release_threshold and tilt < (config.shoulder_tilt_threshold_deg - 1.5)

    if strong_alarm:
        session.alarm_streak += 1
        session.ok_streak = 0
    elif clear_ok:
        session.ok_streak += 1
        session.alarm_streak = 0
    else:
        session.alarm_streak = 0
        session.ok_streak = 0

    if session.alarm_streak >= config.alarm_frames_to_trigger:
        session.stable_status = "ALARM"
    elif session.ok_streak >= config.ok_frames_to_release:
        session.stable_status = "OK"

    result["status"] = session.stable_status
    result["reason"] = "slump_detected" if session.stable_status == "ALARM" else "posture_ok"
    if baseline > 0:
        normalized = float(avg_delta) / baseline
        score = max(0.0, min(100.0, normalized * 100.0))
    else:
        score = 0.0
    if session.stable_status == "ALARM":
        score = min(score, 45.0)
    result["posture_consistency_score"] = round(score, 1)
    result["metrics"]["baseline_delta_px"] = round(baseline, 2)
    result["metrics"]["baseline_drop_threshold_px"] = round(drop_threshold, 2)
    result["metrics"]["baseline_release_threshold_px"] = round(release_threshold, 2)
    result["metrics"]["alarm_streak"] = session.alarm_streak
    result["metrics"]["ok_streak"] = session.ok_streak
    return result


@app.get("/")
async def root() -> HTMLResponse:
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    session = SessionState(
        baseline_samples=[],
        baseline_delta_px=None,
        alarm_streak=0,
        ok_streak=0,
        stable_status="OK",
        prev_keypoints={},
    )
    try:
        while True:
            payload = await websocket.receive_json()
            frame_data = payload.get("frame")
            client_ts = payload.get("client_ts")
            if not frame_data:
                await websocket.send_json(
                    {
                        "status": "NO_POSE",
                        "reason": "missing_frame",
                        "keypoints": {},
                        "posture_consistency_score": 0.0,
                        "metrics": {},
                        "inference_time_ms": 0.0,
                        "client_ts": client_ts,
                        "server_ts": int(time.time() * 1000),
                    }
                )
                continue

            model = get_analyzer()
            if model is None:
                await websocket.send_json(
                    {
                        "status": "NO_POSE",
                        "reason": "mediapipe_init_failed",
                        "keypoints": {},
                        "posture_consistency_score": 0.0,
                        "metrics": {},
                        "inference_time_ms": 0.0,
                        "client_ts": client_ts,
                        "server_ts": int(time.time() * 1000),
                    }
                )
                continue

            result = model.evaluate(frame_data, session.prev_keypoints)
            session.prev_keypoints = result.get("smoothed_keypoints", {})
            result = apply_session_calibration(result, session)
            result.pop("smoothed_keypoints", None)
            result["client_ts"] = client_ts
            result["server_ts"] = int(time.time() * 1000)
            await websocket.send_json(result)
    except WebSocketDisconnect:
        return

