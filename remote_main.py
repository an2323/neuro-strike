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
from dataclasses import dataclass, field
from typing import Dict, Optional, Set
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
    """MediaPipe Pose wrapper using the legacy solutions API (compatible with mediapipe 0.10.x)."""

    def __init__(self, config: Optional[AnalyzerConfig] = None) -> None:
        self.config = config or AnalyzerConfig()
        self.mp_pose = mp.solutions.pose

        # AMD MI300X optimized settings
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # Maximum precision for professional analysis
            smooth_landmarks=True,
            enable_segmentation=False,  # Disable to save GPU memory
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        logger.info("FootballAnalyzer initialized with model_complexity=2")

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
            "inference_time_ms": round((time.perf_counter() - start) * 1000, 2),
            "smoothed_keypoints": keypoints,
            "timestamp_ms": now_ms,
        }

    def close(self):
        """Cleanup MediaPipe resources."""
        if hasattr(self, 'pose'):
            self.pose.close()


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
            "inference_time_ms": 0.0,
            "client_ts": client_ts,
            "frame_seq": frame_seq_in,
            "server_ts": int(time.time() * 1000),
            "source_mode": mode,
        }

    try:
        result = analyzer.evaluate(jpeg_bytes, session.prev_keypoints, session.prev_ts_ms)
        session.prev_keypoints = result.get("smoothed_keypoints", {})
        session.prev_ts_ms = result.get("timestamp_ms")
        result.pop("smoothed_keypoints", None)
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
