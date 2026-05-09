"""
NeuroStrike Football Action Analyzer - Remote Backend for AMD MI300X
Optimized for high-concurrency cloud deployment with connection pooling,
per-session queues, and rate limiting.
"""
import asyncio
import json
import os
import struct
import time
import logging
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from contextlib import asynccontextmanager

# Enable AMD ROCm GPU acceleration for MI300X
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "0")

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_CONCURRENT_SESSIONS = 10
SESSION_QUEUE_MAXSIZE = 8
SESSION_QUEUE_TIMEOUT = 5.0  # seconds
FRAME_RATE_LIMIT = 0.030     # ~33 FPS max per session

POSE_TASK_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
VOICE_NO_PLAYER = "Adjust camera: player not found"

_MODEL_DIR = Path(__file__).resolve().parent / "models"
_POSE_TASK_MODEL_PATH = _MODEL_DIR / "pose_landmarker_lite.task"

# Ankles + foot indices (same as BlazePose / legacy Pose)
_FOOT_LANDMARK_IDX: Tuple[int, ...] = (27, 28, 31, 32)

# Normalized bbox height (max_y - min_y); above this ⇒ foreground / coach-sized — exclude.
POSE_MAX_NORMALIZED_HEIGHT = 0.5
# Sticky re-acquire only after this many consecutive frames without a tracked centroid (~3 s @ 30 fps).
STICKY_SWITCH_AFTER_FRAMES = 90


def _ensure_pose_task_model() -> Path:
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not _POSE_TASK_MODEL_PATH.is_file():
        logger.info("Downloading Pose Landmarker task model to %s", _POSE_TASK_MODEL_PATH)
        urllib.request.urlretrieve(POSE_TASK_MODEL_URL, _POSE_TASK_MODEL_PATH)
    return _POSE_TASK_MODEL_PATH


def _pose_landmark_names() -> Tuple[str, ...]:
    return tuple(
        e.name for e in sorted(mp.solutions.pose.PoseLandmark, key=lambda x: x.value)
    )


def _mean_foot_y_normalized(pose_lms: List) -> float:
    """Mean normalized foot Y; lower => feet higher in frame (often farther subject)."""
    ys: List[float] = []
    for i in _FOOT_LANDMARK_IDX:
        if i < len(pose_lms):
            ys.append(float(pose_lms[i].y))
    return float(sum(ys) / len(ys)) if ys else 1.0


def _pose_bbox_height_norm(pose_lms: List) -> float:
    """Vertical span of the pose in normalized image coords (0–1)."""
    if not pose_lms:
        return 1.0
    ys = [float(lm.y) for lm in pose_lms]
    return float(max(ys) - min(ys)) if ys else 1.0


def _distant_subject_candidates(pose_landmarks: List[List]) -> List[List]:
    """Exclude poses taller than half the frame (e.g. kneeling coach). If all excluded, keep smallest."""
    if not pose_landmarks:
        return []
    small = [p for p in pose_landmarks if _pose_bbox_height_norm(p) <= POSE_MAX_NORMALIZED_HEIGHT]
    if small:
        return small
    return [min(pose_landmarks, key=_pose_bbox_height_norm)]


def _select_by_foot_elevation(pose_landmarks: List[List]) -> List:
    """Fallback: pick pose with feet highest on screen (farther-away player)."""
    if len(pose_landmarks) == 1:
        return pose_landmarks[0]
    best_list: Optional[List] = None
    best_score = float("inf")
    for pose_lms in pose_landmarks:
        score = _mean_foot_y_normalized(pose_lms)
        if score < best_score:
            best_score = score
            best_list = pose_lms
    assert best_list is not None
    return best_list


def _pose_centroid(pose_lms: List) -> Tuple[float, float]:
    """Normalized (x, y) centroid using hip landmarks (indices 23, 24)."""
    HIP_IDX = (23, 24)
    xs: List[float] = []
    ys: List[float] = []
    for i in HIP_IDX:
        if i < len(pose_lms):
            xs.append(float(pose_lms[i].x))
            ys.append(float(pose_lms[i].y))
    if xs:
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    # Ultimate fallback: mean of all landmarks
    all_x = [float(lm.x) for lm in pose_lms]
    all_y = [float(lm.y) for lm in pose_lms]
    if all_x:
        return (sum(all_x) / len(all_x), sum(all_y) / len(all_y))
    return (0.5, 0.5)


def select_sticky_pose(
    pose_landmarks: List[List],
    prev_centroid: Optional[Tuple[float, float]],
    miss_frames: int,
    switch_after: int = STICKY_SWITCH_AFTER_FRAMES,
) -> Tuple[List, Tuple[float, float]]:
    """Sticky-tracking pose selector.

    While the player is continuously visible, re-picks whichever pose stays
    closest to the previous frame's centroid.  Falls back to foot-elevation
    heuristic only when no prior reference exists or the player was missing
    for more than `switch_after` consecutive frames.

    Large foreground figures (normalized bbox height > 50% of frame) are never
    considered for tracking.

    Returns (chosen_landmark_list, centroid_of_chosen).
    """
    if not pose_landmarks:
        raise ValueError("pose_landmarks must be non-empty")
    candidates = _distant_subject_candidates(pose_landmarks)
    if not candidates:
        raise ValueError("pose_landmarks must be non-empty")

    if len(candidates) == 1:
        chosen = candidates[0]
        return chosen, _pose_centroid(chosen)

    # No previous anchor or player was lost too long → reacquire via heuristic
    if prev_centroid is None or miss_frames > switch_after:
        chosen = _select_by_foot_elevation(candidates)
        return chosen, _pose_centroid(chosen)

    # Sticky: pick the pose whose hip centroid is nearest to the last known position
    cx, cy = prev_centroid
    best_list: Optional[List] = None
    best_dist = float("inf")
    for pose_lms in candidates:
        pc = _pose_centroid(pose_lms)
        dist = (pc[0] - cx) ** 2 + (pc[1] - cy) ** 2
        if dist < best_dist:
            best_dist = dist
            best_list = pose_lms
    assert best_list is not None
    return best_list, _pose_centroid(best_list)


def keypoints_from_tasks_pose_list(
    pose_lms: List, width: int, height: int, names: Tuple[str, ...]
) -> Dict[str, Dict[str, float]]:
    points: Dict[str, Dict[str, float]] = {}
    for i, lm in enumerate(pose_lms):
        if i >= len(names):
            break
        vis = float(getattr(lm, "visibility", 1.0) or 1.0)
        points[names[i]] = {
            "x": float(lm.x * width),
            "y": float(lm.y * height),
            "z": float(lm.z),
            "visibility": vis,
        }
    return points


@dataclass
class AnalyzerConfig:
    min_visibility: float = 0.15
    hip_tilt_threshold_deg: float = 14.0
    knee_min_deg: float = 120.0
    knee_max_deg: float = 176.0
    # Hip-relative ankle speed (px/s); whole-body motion (jumps) largely cancels out.
    alarm_speed_threshold_px_s: float = 90.0
    fast_motion_threshold_px: float = 12.0
    # Knee extending toward straight: positive d(angle)/dt (deg/s).
    knee_extension_rate_threshold_deg_s: float = 160.0
    # Ankle must be at least this many pixels below mid-hip (image y grows downward).
    strike_ankle_below_midhip_margin_px: float = 8.0
    # Consecutive frames with strike_candidate before voice_feedback fires.
    voice_strike_consecutive_frames: int = 2


def point_speed_px_per_sec(
    prev_point: Optional[Dict[str, float]], current_point: Optional[Dict[str, float]], dt_sec: float
) -> float:
    if not prev_point or not current_point:
        return 0.0
    dx = current_point["x"] - prev_point["x"]
    dy = current_point["y"] - prev_point["y"]
    return float(np.sqrt(dx * dx + dy * dy) / max(dt_sec, 1e-3))


def _mid_hip_xy(keypoints: Dict[str, Dict[str, float]]) -> Optional[Dict[str, float]]:
    lh = keypoints.get("LEFT_HIP")
    rh = keypoints.get("RIGHT_HIP")
    if not lh or not rh:
        return None
    return {"x": (lh["x"] + rh["x"]) * 0.5, "y": (lh["y"] + rh["y"]) * 0.5}


def _ankle_minus_midhip(keypoints: Dict[str, Dict[str, float]], side: str) -> Optional[Dict[str, float]]:
    m = _mid_hip_xy(keypoints)
    if not m:
        return None
    ank = keypoints.get("LEFT_ANKLE" if side == "LEFT" else "RIGHT_ANKLE")
    if not ank:
        return None
    return {"x": float(ank["x"] - m["x"]), "y": float(ank["y"] - m["y"])}


def relative_ankle_speed_px_per_sec(
    prev_keypoints: Optional[Dict[str, Dict[str, float]]],
    keypoints: Dict[str, Dict[str, float]],
    side: str,
    dt_sec: float,
) -> float:
    """Speed of (ankle - mid_hip) between frames; insensitive to global translation."""
    if not prev_keypoints or dt_sec < 1e-6:
        return 0.0
    v0 = _ankle_minus_midhip(prev_keypoints, side)
    v1 = _ankle_minus_midhip(keypoints, side)
    if not v0 or not v1:
        return 0.0
    dx = v1["x"] - v0["x"]
    dy = v1["y"] - v0["y"]
    return float(np.sqrt(dx * dx + dy * dy) / dt_sec)


def ankle_below_mid_hip(
    keypoints: Dict[str, Dict[str, float]], side: str, margin_px: float
) -> bool:
    """True if kicking ankle is lower on screen than mid-hip (typical strike / leg swing)."""
    m = _mid_hip_xy(keypoints)
    ank = keypoints.get("LEFT_ANKLE" if side == "LEFT" else "RIGHT_ANKLE")
    if not m or not ank:
        return False
    return float(ank["y"]) > float(m["y"]) + float(margin_px)


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

    if fast_motion:
        alpha_prev = 0.35
        alpha_curr = 0.65
    else:
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
    """Pose via MediaPipe Tasks (multi-person) when available; legacy Pose otherwise."""

    def __init__(self, config: Optional[AnalyzerConfig] = None) -> None:
        self.config = config or AnalyzerConfig()
        self.mp_pose = mp.solutions.pose
        self._lm_names = _pose_landmark_names()
        self._use_tasks = False
        self._landmarker = None
        self.pose = None
        self._frame_ts = 0
        self._TaskImage = None
        self._TaskImageFormat = None

        try:
            from mediapipe.tasks.python import vision as tasks_vision
            from mediapipe.tasks.python.core import base_options as task_base
            from mediapipe.tasks.python.vision.core import vision_task_running_mode as vtrm

            model_path = str(_ensure_pose_task_model())
            opts = tasks_vision.PoseLandmarkerOptions(
                base_options=task_base.BaseOptions(model_asset_path=model_path),
                running_mode=vtrm.VisionTaskRunningMode.VIDEO,
                num_poses=4,
                min_pose_detection_confidence=0.25,
                min_pose_presence_confidence=0.25,
                min_tracking_confidence=0.4,
            )
            from mediapipe.tasks.python.vision.core import image as task_image_mod

            self._TaskImage = task_image_mod.Image
            self._TaskImageFormat = task_image_mod.ImageFormat
            self._landmarker = tasks_vision.PoseLandmarker.create_from_options(opts)
            self._use_tasks = True
            logger.info("FootballAnalyzer: PoseLandmarker VIDEO (num_poses=4, conf=0.4)")
        except Exception as e:
            logger.warning("PoseLandmarker unavailable (%s); using legacy Pose", e)
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=0.4,
                min_tracking_confidence=0.4,
            )
            logger.info("FootballAnalyzer: legacy Pose (model_complexity=2, conf=0.4)")

    @staticmethod
    def _decode_frame(jpeg_bytes: bytes) -> Optional[np.ndarray]:
        """Decode raw JPEG bytes directly (no base64)."""
        try:
            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8).copy()
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error(f"Frame decode error: {e}")
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
        jpeg_bytes: bytes,
        prev_keypoints: Optional[Dict[str, Dict[str, float]]] = None,
        prev_ts_ms: Optional[int] = None,
        prev_centroid: Optional[Tuple[float, float]] = None,
        miss_frames: int = 0,
    ) -> Dict:
        start = time.perf_counter()
        now_ms = int(time.time() * 1000)
        frame = self._decode_frame(jpeg_bytes)
        if frame is None:
            return {
                "status": "NO_POSE",
                "reason": "invalid_frame",
                "keypoints": {},
                "kick_speed": 0.0,
                "knee_angle": 180.0,
                "posture_consistency_score": 0.0,
                "metrics": {},
                "voice_feedback": None,
                "inference_time_ms": round((time.perf_counter() - start) * 1000, 2),
                "timestamp_ms": now_ms,
            }

        h, w = frame.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if self._use_tasks and self._landmarker is not None:
            self._frame_ts += 33
            mp_image = self._TaskImage(
                image_format=self._TaskImageFormat.SRGB,
                data=rgb,
            )
            detection_result = self._landmarker.detect_for_video(mp_image, self._frame_ts)
            if not detection_result.pose_landmarks:
                return {
                    "status": "NO_POSE",
                    "reason": "no_pose",
                    "keypoints": {},
                    "kick_speed": 0.0,
                    "knee_angle": 180.0,
                    "posture_consistency_score": 0.0,
                    "metrics": {},
                    "voice_feedback": None,
                    "chosen_centroid": None,
                    "inference_time_ms": round((time.perf_counter() - start) * 1000, 2),
                    "timestamp_ms": now_ms,
                }
            chosen, chosen_centroid = select_sticky_pose(
                detection_result.pose_landmarks, prev_centroid, miss_frames
            )
            raw_keypoints = keypoints_from_tasks_pose_list(chosen, w, h, self._lm_names)
        else:
            assert self.pose is not None
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
                    "voice_feedback": None,
                    "chosen_centroid": None,
                    "inference_time_ms": round((time.perf_counter() - start) * 1000, 2),
                    "timestamp_ms": now_ms,
                }
            raw_keypoints = self._extract_all_keypoints(result.pose_landmarks, w, h)
            # Legacy Pose always returns one person; derive centroid from keypoints
            lh = raw_keypoints.get("LEFT_HIP")
            rh = raw_keypoints.get("RIGHT_HIP")
            chosen_centroid: Tuple[float, float]
            if lh and rh and w > 0 and h > 0:
                chosen_centroid = (
                    (lh["x"] / w + rh["x"] / w) * 0.5,
                    (lh["y"] / h + rh["y"] / h) * 0.5,
                )
            else:
                chosen_centroid = (0.5, 0.5)
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
                "voice_feedback": None,
                "chosen_centroid": None,
                "smoothed_keypoints": keypoints,
                "inference_time_ms": round((time.perf_counter() - start) * 1000, 2),
                "timestamp_ms": now_ms,
            }

        dt_sec = ((now_ms - prev_ts_ms) / 1000.0) if prev_ts_ms else (1.0 / 30.0)
        dt_sec = max(1e-3, dt_sec)

        left_rel = relative_ankle_speed_px_per_sec(prev_keypoints, keypoints, "LEFT", dt_sec)
        right_rel = relative_ankle_speed_px_per_sec(prev_keypoints, keypoints, "RIGHT", dt_sec)
        left_abs = point_speed_px_per_sec(
            prev_keypoints.get("LEFT_ANKLE") if prev_keypoints else None, keypoints.get("LEFT_ANKLE"), dt_sec
        )
        right_abs = point_speed_px_per_sec(
            prev_keypoints.get("RIGHT_ANKLE") if prev_keypoints else None, keypoints.get("RIGHT_ANKLE"), dt_sec
        )
        kicking_leg = "LEFT" if left_rel >= right_rel else "RIGHT"
        kick_speed = left_rel if kicking_leg == "LEFT" else right_rel

        if kicking_leg == "LEFT":
            hip, knee, ankle = keypoints.get("LEFT_HIP"), keypoints.get("LEFT_KNEE"), keypoints.get("LEFT_ANKLE")
        else:
            hip, knee, ankle = keypoints.get("RIGHT_HIP"), keypoints.get("RIGHT_KNEE"), keypoints.get("RIGHT_ANKLE")
        knee_angle = self._joint_angle_deg(hip, knee, ankle) if hip and knee and ankle else 180.0

        def _leg_knee_angle_deg(kp: Dict[str, Dict[str, float]], side: str) -> float:
            if side == "LEFT":
                h, k, a = kp.get("LEFT_HIP"), kp.get("LEFT_KNEE"), kp.get("LEFT_ANKLE")
            else:
                h, k, a = kp.get("RIGHT_HIP"), kp.get("RIGHT_KNEE"), kp.get("RIGHT_ANKLE")
            return self._joint_angle_deg(h, k, a) if h and k and a else 180.0

        prev_knee: Optional[float] = None
        if prev_keypoints:
            prev_knee = _leg_knee_angle_deg(prev_keypoints, kicking_leg)
        knee_extension_rate = ((knee_angle - prev_knee) / dt_sec) if prev_knee is not None else 0.0

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

        thresh = self.config.alarm_speed_threshold_px_s
        speed_ok = prev_keypoints is not None and kick_speed > thresh
        vertical_ok = ankle_below_mid_hip(
            keypoints, kicking_leg, self.config.strike_ankle_below_midhip_margin_px
        )
        extension_ok = (
            prev_knee is not None
            and knee_extension_rate >= self.config.knee_extension_rate_threshold_deg_s
        )
        strike_candidate = bool(speed_ok and vertical_ok and extension_ok)

        voice_feedback_raw: Optional[str] = None
        if strike_candidate:
            if hip_tilt_deg > self.config.hip_tilt_threshold_deg:
                voice_feedback_raw = "Lean forward"
            elif knee_angle > self.config.knee_max_deg:
                voice_feedback_raw = "Bend your knee"
            else:
                voice_feedback_raw = "Great strike"

        print(
            f"Debug: Speed={kick_speed:.1f} Ext={knee_extension_rate:.1f} "
            f"Found=True SpeedOK={speed_ok} VertOK={vertical_ok} ExtOK={extension_ok}"
        )

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
                "left_ankle_speed_px_s": round(float(left_rel), 2),
                "right_ankle_speed_px_s": round(float(right_rel), 2),
                "left_ankle_abs_speed_px_s": round(float(left_abs), 2),
                "right_ankle_abs_speed_px_s": round(float(right_abs), 2),
                "knee_extension_rate_deg_s": round(float(knee_extension_rate), 1),
                "strike_speed_ok": speed_ok,
                "strike_vertical_ok": vertical_ok,
                "strike_extension_ok": extension_ok,
                "strike_candidate": strike_candidate,
                "fast_motion_mode": fast_motion,
            },
            "voice_feedback": None,
            "voice_feedback_raw": voice_feedback_raw,
            "chosen_centroid": chosen_centroid,
            "inference_time_ms": round((time.perf_counter() - start) * 1000, 2),
            "smoothed_keypoints": keypoints,
            "timestamp_ms": now_ms,
        }

    def close(self) -> None:
        if getattr(self, "_landmarker", None) is not None:
            self._landmarker.close()
            self._landmarker = None
        if getattr(self, "pose", None) is not None:
            self.pose.close()
            self.pose = None


# ---------------------------------------------------------------------------
# Connection pool & session management
# ---------------------------------------------------------------------------

@dataclass
class SessionState:
    prev_keypoints: Dict[str, Dict[str, float]]
    prev_ts_ms: Optional[int]
    mode: str
    connection_id: str
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=SESSION_QUEUE_MAXSIZE))
    last_frame_time: float = 0.0
    strike_voice_streak: int = 0
    sticky_centroid: Optional[Tuple[float, float]] = None
    sticky_miss_frames: int = 0
    no_pose_streak: int = 0


class ConnectionPool:
    """Manages concurrent WebSocket sessions with a hard limit."""

    def __init__(self, max_sessions: int = MAX_CONCURRENT_SESSIONS):
        self.max_sessions = max_sessions
        self._sessions: Dict[str, SessionState] = {}
        self._lock = asyncio.Lock()

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    async def register(self, connection_id: str) -> bool:
        async with self._lock:
            if len(self._sessions) >= self.max_sessions:
                logger.warning(f"Connection limit reached ({self.max_sessions}). Rejecting {connection_id}")
                return False
            self._sessions[connection_id] = SessionState(
                prev_keypoints={},
                prev_ts_ms=None,
                mode="WEBCAM",
                connection_id=connection_id,
            )
            logger.info(f"Session registered: {connection_id} (active: {len(self._sessions)})")
            return True

    async def unregister(self, connection_id: str) -> None:
        async with self._lock:
            self._sessions.pop(connection_id, None)
            logger.info(f"Session unregistered: {connection_id} (active: {len(self._sessions)})")

    def get(self, connection_id: str) -> Optional[SessionState]:
        return self._sessions.get(connection_id)


# Global instances
analyzer: Optional[FootballAnalyzer] = None
connection_pool = ConnectionPool(max_sessions=MAX_CONCURRENT_SESSIONS)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    global analyzer
    logger.info("Initializing NeuroStrike Remote Backend (AMD MI300X)...")
    try:
        analyzer = FootballAnalyzer()
        logger.info("✓ MediaPipe Pose initialized successfully (GPU: ROCm)")
    except Exception as e:
        logger.error(f"✗ Failed to initialize MediaPipe: {e}")
        analyzer = None

    yield

    # Cleanup
    logger.info("Shutting down NeuroStrike Remote Backend...")
    if analyzer:
        analyzer.close()
    logger.info("Shutdown complete.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="NeuroStrike Remote Backend (AMD MI300X)",
    description="High-performance football biomechanics analysis with connection pooling",
    version="2.1.0",
    lifespan=lifespan
)

# CORS configuration for remote access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """Health check endpoint."""
    return JSONResponse({
        "service": "NeuroStrike Remote Backend",
        "status": "online",
        "model_complexity": 2,
        "active_connections": connection_pool.active_count,
        "max_connections": MAX_CONCURRENT_SESSIONS,
        "gpu_enabled": os.environ.get("MEDIAPIPE_DISABLE_GPU", "1") == "0",
        "gpu_backend": "ROCm (AMD MI300X)",
    })


@app.get("/health")
async def health_check():
    """Detailed health check."""
    return JSONResponse({
        "status": "healthy" if analyzer else "degraded",
        "analyzer_ready": analyzer is not None,
        "active_sessions": connection_pool.active_count,
        "max_sessions": MAX_CONCURRENT_SESSIONS,
        "timestamp": int(time.time() * 1000),
    })


# ---------------------------------------------------------------------------
# WebSocket endpoint with high-concurrency support
# ---------------------------------------------------------------------------

async def process_frame(
    session: SessionState,
    jpeg_bytes: bytes,
    client_ts: Optional[int],
    frame_seq_in: Optional[int],
    mode: str,
) -> Dict:
    """Run inference on a single frame with rate limiting."""
    now = time.monotonic()
    elapsed = now - session.last_frame_time
    if elapsed < FRAME_RATE_LIMIT:
        await asyncio.sleep(FRAME_RATE_LIMIT - elapsed)
    session.last_frame_time = time.monotonic()

    if analyzer is None:
        return {
            "status": "NO_POSE",
            "reason": "mediapipe_init_failed",
            "keypoints": {},
            "kick_speed": 0.0,
            "knee_angle": 180.0,
            "posture_consistency_score": 0.0,
            "metrics": {},
            "voice_feedback": None,
            "inference_time_ms": 0.0,
            "client_ts": client_ts,
            "frame_seq": frame_seq_in,
            "server_ts": int(time.time() * 1000),
            "source_mode": mode,
        }

    if not jpeg_bytes:
        return {
            "status": "NO_POSE",
            "reason": "missing_frame",
            "keypoints": {},
            "kick_speed": 0.0,
            "knee_angle": 180.0,
            "posture_consistency_score": 0.0,
            "metrics": {},
            "voice_feedback": None,
            "inference_time_ms": 0.0,
            "client_ts": client_ts,
            "frame_seq": frame_seq_in,
            "server_ts": int(time.time() * 1000),
            "source_mode": mode,
        }

    try:
        result = analyzer.evaluate(
            jpeg_bytes,
            session.prev_keypoints,
            session.prev_ts_ms,
            prev_centroid=session.sticky_centroid,
            miss_frames=session.sticky_miss_frames,
        )
        session.prev_keypoints = result.get("smoothed_keypoints", {})
        session.prev_ts_ms = result.get("timestamp_ms")
        result.pop("smoothed_keypoints", None)

        # ---- Sticky tracking state update ----
        new_centroid = result.pop("chosen_centroid", None)
        if new_centroid is not None:
            session.sticky_centroid = new_centroid
            session.sticky_miss_frames = 0
        else:
            session.sticky_miss_frames += 1

        # ---- Strike voice debounce (2 consecutive frames) ----
        raw_phrase = result.pop("voice_feedback_raw", None)
        metrics = result.get("metrics") or {}
        if metrics.get("strike_candidate"):
            session.strike_voice_streak += 1
        else:
            session.strike_voice_streak = 0

        if raw_phrase is not None:
            need = analyzer.config.voice_strike_consecutive_frames
            result["voice_feedback"] = raw_phrase if session.strike_voice_streak == need else None

        # ---- NO_POSE voice gating: only speak after 45 consecutive missing frames ----
        if result.get("status") == "NO_POSE":
            session.no_pose_streak += 1
            if result.get("voice_feedback") is None and session.no_pose_streak > 20:
                result["voice_feedback"] = VOICE_NO_PLAYER
        else:
            session.no_pose_streak = 0

        result["client_ts"] = client_ts
        result["frame_seq"] = frame_seq_in
        result["server_ts"] = int(time.time() * 1000)
        result["source_mode"] = mode
        return result
    except Exception as e:
        logger.error(f"Analysis error for {session.connection_id}: {e}")
        return {
            "status": "ERROR",
            "reason": "analysis_failed",
            "keypoints": {},
            "kick_speed": 0.0,
            "knee_angle": 180.0,
            "posture_consistency_score": 0.0,
            "metrics": {},
            "voice_feedback": None,
            "inference_time_ms": 0.0,
            "client_ts": client_ts,
            "frame_seq": frame_seq_in,
            "server_ts": int(time.time() * 1000),
            "source_mode": mode,
        }


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    connection_id = f"{websocket.client.host}:{websocket.client.port}_{int(time.time() * 1000)}"

    # --- Connection limit check ---
    if not await connection_pool.register(connection_id):
        await websocket.close(code=1013, reason="Server at capacity. Try again later.")
        return

    try:
        await websocket.accept()
        logger.info(f"✓ WebSocket connected: {connection_id}")

        session = connection_pool.get(connection_id)
        if session is None:
            await websocket.close(code=1011, reason="Session initialization failed")
            return

        # Background consumer: reads from session queue and sends results
        async def queue_consumer():
            try:
                while True:
                    result = await session.queue.get()
                    try:
                        await websocket.send_json(result)
                    except Exception:
                        break
                    finally:
                        session.queue.task_done()
            except asyncio.CancelledError:
                pass

        consumer_task = asyncio.create_task(queue_consumer())

        try:
            while True:
                # Receive with timeout to detect stale connections
                try:
                    raw_msg = await asyncio.wait_for(
                        websocket.receive_bytes(),
                        timeout=SESSION_QUEUE_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Receive timeout for {connection_id}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to receive binary from {connection_id}: {e}")
                    break

                # Parse binary protocol: [4-byte header_len][JSON header][JPEG bytes]
                if len(raw_msg) < 5:
                    logger.warning(f"Malformed binary message from {connection_id} (too short)")
                    break
                header_len = struct.unpack_from('<I', raw_msg, 0)[0]
                if 4 + header_len > len(raw_msg):
                    logger.warning(f"Malformed binary message from {connection_id} (header truncated)")
                    break
                header = json.loads(raw_msg[4:4 + header_len].decode('utf-8'))
                jpeg_bytes = raw_msg[4 + header_len:]

                client_ts = header.get("client_ts")
                frame_seq_in = header.get("frame_seq")
                mode = str(header.get("mode", session.mode)).upper()
                session.mode = mode

                # Process frame and enqueue result
                result = await process_frame(session, jpeg_bytes, client_ts, frame_seq_in, mode)

                # If queue is full, drop oldest frame to keep latency low
                if session.queue.full():
                    try:
                        session.queue.get_nowait()
                        session.queue.task_done()
                    except asyncio.QueueEmpty:
                        pass

                try:
                    session.queue.put_nowait(result)
                except asyncio.QueueFull:
                    logger.warning(f"Dropping frame for {connection_id} (queue full)")

        except WebSocketDisconnect:
            logger.info(f"✗ WebSocket disconnected: {connection_id}")
        except Exception as e:
            logger.error(f"WebSocket error for {connection_id}: {e}")
        finally:
            consumer_task.cancel()
            try:
                await consumer_task
            except asyncio.CancelledError:
                pass

    finally:
        await connection_pool.unregister(connection_id)
        logger.info(f"Session cleaned up: {connection_id} (Active: {connection_pool.active_count})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info",
        access_log=True,
        ws_ping_interval=20,
        ws_ping_timeout=20,
        workers=1,  # Single worker; MediaPipe is not fork-safe
    )
