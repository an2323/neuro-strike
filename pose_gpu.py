"""
pose_gpu.py — BlazePose ONNX inference on AMD MI300X via onnxruntime MIGraphX EP.

Implements the identical two-stage pipeline as mediapipe.solutions.pose.Pose:
  Stage 1 — Pose detector (128×128 input):
      Generates 896 SSD anchors, decodes boxes, NMS → rotated body ROI.
  Stage 2 — Pose landmark model (256×256 rotated crop):
      33 body landmarks in crop space → reprojected to full image.

Output API mirrors mediapipe.solutions.pose.Pose.process():
    result = estimator.process(rgb_frame)
    result.pose_landmarks.landmark[i].x / .y / .z / .visibility

Callers (strike_video_processor.py, remote_main.py) need only a one-line
change to use this instead of mediapipe — the rest of the biomechanics code
is untouched because all 33 BlazePose keypoints are preserved exactly.

Provider priority: MIGraphXExecutionProvider → ROCMExecutionProvider → CPU.
Falls back gracefully to CPU if ROCm is unavailable.

ONNX models required in models/:
  pose_detection.onnx      — 128×128 person detector (896 anchors)
  pose_landmark_heavy.onnx — 256×256 landmark model (complexity=2 equivalent)

Run scripts/download_blazepose_onnx.sh to fetch these before first use.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_MODEL_DIR = Path(__file__).resolve().parent / "models"
_DETECTOR_TFLITE_PATH = _MODEL_DIR / "pose_detector.tflite"
_LANDMARK_PATH = _MODEL_DIR / "pose_landmark_heavy.onnx"

# ---------------------------------------------------------------------------
# Constants matching MediaPipe BlazePose configuration
# ---------------------------------------------------------------------------
# mediapipe 0.10+ ships the 224×224 detector (2254 SSD anchors, strides 8/16/32)
_DET_INPUT = 224          # detector square input size
_LM_INPUT  = 256          # landmark model square input size
_SSD_SCALE = 224.0        # SSD box decode scale (x_scale = y_scale = w_scale = h_scale)
_SCORE_THRESH   = 0.3     # min detection confidence
_NMS_IOU_THRESH = 0.3     # NMS overlap threshold
_ROI_SCALE      = 2.6     # MediaPipe RectTransformationCalculator scale
_ROI_SHIFT      = 0.1     # shift center towards hips for full body coverage
_POSE_FLAG_THRESH = 0.5   # min landmark model "pose present" score
_TRACK_MAX_CENTER_JUMP = 0.20  # normalized image units; prevents switching people
_TRACK_MAX_HEIGHT_RATIO = 2.2
_TRACK_MIN_HEIGHT_RATIO = 0.45
_MAX_ROI_TRACK_FRAMES = 45

# mediapipe 0.10+ landmark model outputs 39 landmarks (33 body + 6 auxiliary).
# We keep only the first 33 which map to the canonical BlazePose keypoint set.
_LM_TOTAL = 39            # total landmarks from model
_LM_BODY  = 33            # body landmarks exposed in the public API


# ---------------------------------------------------------------------------
# MediaPipe-compatible result objects (duck-typing)
# ---------------------------------------------------------------------------

class _Landmark:
    __slots__ = ("x", "y", "z", "visibility", "presence")

    def __init__(
        self,
        x: float,
        y: float,
        z: float,
        visibility: float,
        presence: float = 1.0,
    ) -> None:
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.visibility = float(visibility)
        self.presence = float(presence)


class _PoseLandmarkList:
    def __init__(self, landmarks: List[_Landmark]) -> None:
        self.landmark: List[_Landmark] = landmarks


class _PoseResult:
    """Return type matching mediapipe.solutions.pose.Pose.process()."""

    def __init__(self, landmarks: Optional[List[_Landmark]]) -> None:
        self.pose_landmarks: Optional[_PoseLandmarkList] = (
            _PoseLandmarkList(landmarks) if landmarks else None
        )


# ---------------------------------------------------------------------------
# SSD anchor generation
# ---------------------------------------------------------------------------

def _make_anchors() -> np.ndarray:
    """
    Generate 2254 SSD anchors for the mediapipe 0.10+ BlazePose pose detector.

    MediaPipe SsdAnchorsCalculator config for the 224×224 model:
      strides:           [8, 16, 32]
      anchors_per_cell:  [2, 2, 6]   (6 at stride 32 from 3 aspect ratios × 2)
      fixed_anchor_size: true
      Count: 28×28×2 + 14×14×2 + 7×7×6 = 1568 + 392 + 294 = 2254

    Returns float32 [2254, 2] — (cx, cy) in normalised [0, 1] image coords.
    """
    anchors: list = []
    for stride, n_per_cell in [(8, 2), (16, 2), (32, 6)]:
        grid = _DET_INPUT // stride
        for y in range(grid):
            for x in range(grid):
                cx = (x + 0.5) / grid
                cy = (y + 0.5) / grid
                for _ in range(n_per_cell):
                    anchors.append([cx, cy])
    result = np.array(anchors, dtype=np.float32)
    assert result.shape == (2254, 2), f"Anchor count mismatch: {result.shape}"
    return result


_ANCHORS: np.ndarray = _make_anchors()


# ---------------------------------------------------------------------------
# SSD decoding helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x.astype(np.float32), -88.0, 88.0)))


def _decode_boxes(raw: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """
    Decode raw SSD box tensor [N, 12] to absolute normalised coords.

    MediaPipe pose detector uses reverse_output_order=true:
      [0] cx_delta, [1] cy_delta, [2] w, [3] h,
      [4,5] kp0(x,y), [6,7] kp1(x,y), [8,9] kp2(x,y), [10,11] kp3(x,y)

    Output layout is [cy, cx, h, w, kp0_y, kp0_x, ...] for downstream code.
    ``anchors`` stores [cx, cy] and must match the leading dimension of ``raw``.
    """
    boxes = np.empty_like(raw)
    boxes[:, 0] = anchors[:, 1] + raw[:, 1] / _SSD_SCALE   # cy
    boxes[:, 1] = anchors[:, 0] + raw[:, 0] / _SSD_SCALE   # cx
    boxes[:, 2] = raw[:, 3] / _SSD_SCALE                    # h
    boxes[:, 3] = raw[:, 2] / _SSD_SCALE                    # w
    for k in range(4):
        b = 4 + k * 2
        boxes[:, b]     = anchors[:, 1] + raw[:, b + 1] / _SSD_SCALE  # ky
        boxes[:, b + 1] = anchors[:, 0] + raw[:, b]     / _SSD_SCALE  # kx
    return boxes


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union for two [cy, cx, h, w] centre-format boxes."""
    ay0, ax0 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ay1, ax1 = a[0] + a[2] / 2, a[1] + a[3] / 2
    by0, bx0 = b[0] - b[2] / 2, b[1] - b[3] / 2
    by1, bx1 = b[0] + b[2] / 2, b[1] + b[3] / 2
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    inter = ih * iw
    union = a[2] * a[3] + b[2] * b[3] - inter
    return float(inter / union) if union > 1e-8 else 0.0


def _nms(boxes: np.ndarray, scores: np.ndarray) -> List[int]:
    order = np.argsort(scores)[::-1].tolist()
    kept: List[int] = []
    while order:
        i = order.pop(0)
        kept.append(i)
        order = [j for j in order if _box_iou(boxes[i], boxes[j]) < _NMS_IOU_THRESH]
    return kept


def _best_detection(
    raw_boxes: np.ndarray, raw_scores: np.ndarray
) -> Optional[np.ndarray]:
    """Return best decoded detection [12] or None if nothing above threshold."""
    candidates = _candidate_detections(raw_boxes, raw_scores)
    return candidates[0][0] if candidates else None


def _candidate_detections(
    raw_boxes: np.ndarray, raw_scores: np.ndarray
) -> List[Tuple[np.ndarray, float]]:
    """Return decoded NMS detections sorted by confidence."""
    scores = _sigmoid(raw_scores.reshape(-1))
    mask = scores >= _SCORE_THRESH
    if not mask.any():
        return []
    idx = np.where(mask)[0]
    decoded = _decode_boxes(raw_boxes[idx], _ANCHORS[idx])
    kept = _nms(decoded, scores[idx])
    return [(decoded[i], float(scores[idx[i]])) for i in kept]


# ---------------------------------------------------------------------------
# ROI geometry
# ---------------------------------------------------------------------------

def _rotation_from_detection(det: np.ndarray) -> float:
    """
    Estimate body rotation angle from the first two detector keypoints.

    MediaPipe uses kp0 = mid-hips, kp1 = mid-shoulders to define the vertical
    body axis.  The target rotation aligns this axis vertically so the landmark
    crop contains an upright person.

    det layout: [cy, cx, h, w, kp0_y, kp0_x, kp1_y, kp1_x, ...]
    """
    kp0_y, kp0_x = float(det[4]), float(det[5])
    kp1_y, kp1_x = float(det[6]), float(det[7])
    dy = kp1_y - kp0_y
    dx = kp1_x - kp0_x
    angle = math.atan2(dx, -dy)   # 0 = body vertical
    return float(np.clip(angle, -math.pi / 2, math.pi / 2))


def _detection_track_state(det: np.ndarray) -> Tuple[np.ndarray, float]:
    """Return a stable normalized center/height estimate for subject tracking."""
    kp0_y, kp0_x = float(det[4]), float(det[5])
    kp1_y, kp1_x = float(det[6]), float(det[7])
    center = np.array(
        [
            (kp0_x + kp1_x) / 2 + _ROI_SHIFT * (kp0_x - kp1_x),
            (kp0_y + kp1_y) / 2 + _ROI_SHIFT * (kp0_y - kp1_y),
        ],
        dtype=np.float32,
    )
    height = max(abs(kp0_y - kp1_y), float(det[2]), 1e-6)
    return center, height


def _landmark_track_state(landmarks: List[_Landmark]) -> Tuple[Optional[np.ndarray], float]:
    """Return normalized center/height from final landmarks."""
    idxs = (11, 12, 23, 24, 25, 26, 27, 28)
    pts = [
        np.array([landmarks[i].x, landmarks[i].y], dtype=np.float32)
        for i in idxs
        if i < len(landmarks)
        and landmarks[i].visibility >= 0.2
        and np.isfinite(landmarks[i].x)
        and np.isfinite(landmarks[i].y)
    ]
    if len(pts) < 3:
        return None, 0.0
    arr = np.stack(pts, axis=0)
    return np.mean(arr, axis=0), float(np.max(arr[:, 1]) - np.min(arr[:, 1]))


def _select_tracked_detection(
    candidates: List[Tuple[np.ndarray, float]],
    prev_center: Optional[np.ndarray],
    prev_height: Optional[float],
) -> Optional[np.ndarray]:
    """Choose the same athlete across frames instead of the highest-score person."""
    if not candidates:
        return None
    if prev_center is None or prev_height is None or prev_height <= 1e-6:
        return candidates[0][0]

    best: Optional[Tuple[float, np.ndarray]] = None
    max_jump = max(_TRACK_MAX_CENTER_JUMP, prev_height * 1.25)
    for det, score in candidates:
        center, height = _detection_track_state(det)
        jump = float(np.linalg.norm(center - prev_center))
        ratio = height / prev_height
        if jump > max_jump:
            continue
        if ratio > _TRACK_MAX_HEIGHT_RATIO or ratio < _TRACK_MIN_HEIGHT_RATIO:
            continue
        # Prefer proximity first, confidence second.
        cost = jump - 0.03 * score
        if best is None or cost < best[0]:
            best = (cost, det)
    return best[1] if best is not None else None


def _roi_from_detection(
    det: np.ndarray, img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """
    Convert a decoded detection to a rotated square crop descriptor
    using the MediaPipe keypoint-based ROI algorithm.

    kp0 = mid-hips, kp1 = mid-shoulders.
    ROI center is the midpoint shifted towards hips for full body coverage.
    ROI size is proportional to the hip–shoulder distance × _ROI_SCALE.

    Returns (cx_px, cy_px, size_px, angle_rad).
    """
    kp0_y, kp0_x = float(det[4]), float(det[5])
    kp1_y, kp1_x = float(det[6]), float(det[7])

    cx_n = (kp0_x + kp1_x) / 2 + _ROI_SHIFT * (kp0_x - kp1_x)
    cy_n = (kp0_y + kp1_y) / 2 + _ROI_SHIFT * (kp0_y - kp1_y)

    kp_dist_px = math.hypot((kp0_x - kp1_x) * img_w, (kp0_y - kp1_y) * img_h)
    size_px = kp_dist_px * _ROI_SCALE

    cx_px = cx_n * img_w
    cy_px = cy_n * img_h
    angle = _rotation_from_detection(det)
    return cx_px, cy_px, size_px, angle


def _roi_from_landmarks(
    landmarks: List[_Landmark], img_w: int, img_h: int
) -> Optional[Tuple[float, float, float, float]]:
    """Build the next-frame ROI from tracked landmarks, similar to MediaPipe tracking mode."""
    required = (11, 12, 23, 24)
    if any(i >= len(landmarks) for i in required):
        return None
    if any(landmarks[i].visibility < 0.15 for i in required):
        return None
    pts = np.array([[landmarks[i].x, landmarks[i].y] for i in range(min(len(landmarks), 33))], dtype=np.float32)
    if not np.all(np.isfinite(pts[list(required)])):
        return None

    mid_hip = (pts[23] + pts[24]) * 0.5
    mid_sh = (pts[11] + pts[12]) * 0.5
    axis = mid_sh - mid_hip
    axis_len_px = math.hypot(float(axis[0]) * img_w, float(axis[1]) * img_h)
    if axis_len_px < 8.0:
        return None

    center_n = (mid_hip + mid_sh) * 0.5 + _ROI_SHIFT * (mid_hip - mid_sh)
    angle = math.atan2(float(axis[0]), -float(axis[1]))
    angle = float(np.clip(angle, -math.pi / 2, math.pi / 2))

    body_idxs = [11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    visible = [
        i
        for i in body_idxs
        if i < len(landmarks)
        and landmarks[i].visibility >= 0.15
        and np.all(np.isfinite(pts[i]))
    ]
    if len(visible) >= 4:
        arr = pts[visible]
        span_px = max(
            float(np.ptp(arr[:, 0]) * img_w),
            float(np.ptp(arr[:, 1]) * img_h),
            axis_len_px,
        )
    else:
        span_px = axis_len_px
    size_px = max(axis_len_px * _ROI_SCALE, span_px * 1.8, 64.0)
    return float(center_n[0] * img_w), float(center_n[1] * img_h), float(size_px), angle


# ---------------------------------------------------------------------------
# Affine crop / unproject
# ---------------------------------------------------------------------------

def _build_affine(
    cx: float, cy: float, size: float, angle: float, out: int = _LM_INPUT
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build affine matrices M (image → crop) and M_inv (crop → image).

    The crop is a rotated square of side `size` centred at (cx, cy).
    M maps image pixel coords to 256×256 crop coords.
    """
    s = out / size
    ca, sa = math.cos(-angle), math.sin(-angle)
    M = np.array(
        [
            [s * ca, -s * sa, out / 2 - s * (ca * cx - sa * cy)],
            [s * sa,  s * ca, out / 2 - s * (sa * cx + ca * cy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return M, np.linalg.inv(M)


def _crop_roi(
    rgb: np.ndarray, cx: float, cy: float, size: float, angle: float
) -> np.ndarray:
    """Warp-affine ROI crop → float32 [0,1] RGB, shape (256, 256, 3)."""
    M, _ = _build_affine(cx, cy, size, angle)
    crop = cv2.warpAffine(
        rgb,
        M[:2],
        (_LM_INPUT, _LM_INPUT),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return crop.astype(np.float32) / 255.0


def _unproject(
    lm_raw: np.ndarray,   # [33, ≥4] — x,y in crop pixels or normalised
    M_inv: np.ndarray,
    img_w: int,
    img_h: int,
) -> List[_Landmark]:
    """
    Map 33 landmarks from crop space back to normalised full-image coords.

    Handles two common ONNX output conventions:
      • Pixel coords in [0, 256]  → apply affine unproject
      • Already normalised [0, 1] → scale to crop pixels first
    """
    # Detect convention: if max(x) ≤ 2 the model outputs normalised coords
    x_max = float(np.max(np.abs(lm_raw[:, 0])))
    already_norm = x_max <= 2.0
    scale = float(_LM_INPUT) if already_norm else 1.0

    # Visibility: apply sigmoid only if values look like raw logits (|v| > 1)
    vis_col = lm_raw[:, 3].copy()
    if float(np.max(np.abs(vis_col))) > 1.5:
        vis_col = _sigmoid(vis_col)

    pres_col = lm_raw[:, 4].copy() if lm_raw.shape[1] > 4 else np.ones(33, np.float32)
    if float(np.max(np.abs(pres_col))) > 1.5:
        pres_col = _sigmoid(pres_col)

    lms: List[_Landmark] = []
    for i in range(33):
        xc = float(lm_raw[i, 0]) * scale
        yc = float(lm_raw[i, 1]) * scale
        zc = float(lm_raw[i, 2]) * scale
        pt = M_inv @ np.array([xc, yc, 1.0])
        x_n = float(np.clip(pt[0] / img_w, 0.0, 1.0))
        y_n = float(np.clip(pt[1] / img_h, 0.0, 1.0))
        z_n = zc / _LM_INPUT   # relative depth in normalised units
        lms.append(_Landmark(x_n, y_n, z_n, float(vis_col[i]), float(pres_col[i])))
    return lms


# ---------------------------------------------------------------------------
# Main estimator
# ---------------------------------------------------------------------------

class BlazePoseONNX:
    """
    Drop-in replacement for mediapipe.solutions.pose.Pose using ONNX Runtime.

    Provides:
      process(rgb)            → _PoseResult   (single frame, real-time path)
      estimate_batch(frames)  → List[_PoseResult]  (offline batch path)

    Provider selection:
      MIGraphXExecutionProvider (AMD MI300X) if available,
      else ROCMExecutionProvider, else CPUExecutionProvider.
    """

    def __init__(
        self,
        detector_path: Optional[str] = None,
        landmark_path: Optional[str] = None,
    ) -> None:
        import onnxruntime as ort

        det_p = Path(detector_path or _DETECTOR_TFLITE_PATH)
        lm_p  = Path(landmark_path or _LANDMARK_PATH)
        if not det_p.is_file():
            raise FileNotFoundError(
                f"TFLite detector not found: {det_p}\n"
                "Download: curl -o models/pose_detector.tflite "
                "https://storage.googleapis.com/mediapipe-assets/pose_detection.tflite"
            )
        if not lm_p.is_file():
            raise FileNotFoundError(
                f"ONNX landmark model not found: {lm_p}\n"
                "Run: bash scripts/download_blazepose_onnx.sh"
            )

        # Detector: TFLite (ONNX conversion breaks DENSIFY sparse tensors)
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            try:
                import tflite_runtime.interpreter as _tfl
                Interpreter = _tfl.Interpreter
            except ImportError:
                import tensorflow as tf
                Interpreter = tf.lite.Interpreter
        self._det_interp = Interpreter(model_path=str(det_p))
        self._det_interp.allocate_tensors()
        self._det_inp = self._det_interp.get_input_details()
        self._det_out = self._det_interp.get_output_details()

        # Landmark: ONNX via MIGraphX / ROCm / CPU
        providers = _pick_providers(ort)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._lm = ort.InferenceSession(str(lm_p), sess_options=opts, providers=providers)
        self._lm_in_name = self._lm.get_inputs()[0].name

        active = self._lm.get_providers()
        self.using_migraphx = "MIGraphXExecutionProvider" in active
        logger.info(
            "BlazePoseONNX ready — providers=%s  migraphx=%s",
            active,
            self.using_migraphx,
        )

    # ------------------------------------------------------------------
    # Internal inference steps
    # ------------------------------------------------------------------

    def _detect_candidates(self, rgb: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """Run TFLite person detector → decoded detections sorted by confidence."""
        inp = cv2.resize(rgb, (_DET_INPUT, _DET_INPUT)).astype(np.float32) / 255.0
        self._det_interp.set_tensor(self._det_inp[0]['index'], inp[np.newaxis])
        self._det_interp.invoke()
        raw_boxes  = np.squeeze(self._det_interp.get_tensor(self._det_out[0]['index']))
        raw_scores = np.squeeze(self._det_interp.get_tensor(self._det_out[1]['index']))
        if raw_scores.ndim == 1:
            raw_scores = raw_scores[:, np.newaxis]
        return _candidate_detections(raw_boxes, raw_scores)

    def _detect(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        """Run TFLite person detector → best decoded detection [12] or None."""
        candidates = self._detect_candidates(rgb)
        return candidates[0][0] if candidates else None

    def _landmark(self, crop: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        Run landmark model on 256×256 float32 [0,1] RGB crop.
        Returns (lm_raw [33, ≥4], pose_flag) or (None, 0.0).
        """
        outs = self._lm.run(None, {self._lm_in_name: crop[np.newaxis]})

        # Output 0: landmarks — mediapipe 0.10+ gives [1,195] (39 lm × 5)
        # older models may give [1,165] (33×5) or [1,33,5].
        lm_raw = np.squeeze(outs[0]).astype(np.float32)
        if lm_raw.ndim == 1:
            total = lm_raw.shape[0]
            # Determine values-per-landmark: try 39-landmark layout first, then 33
            if total % _LM_TOTAL == 0:
                n_vals = total // _LM_TOTAL
                lm_raw = lm_raw.reshape(_LM_TOTAL, n_vals)
            elif total % _LM_BODY == 0:
                n_vals = total // _LM_BODY
                lm_raw = lm_raw.reshape(_LM_BODY, n_vals)
            else:
                lm_raw = lm_raw.reshape(-1, 1)
        # Keep only the first _LM_BODY (33) body landmarks
        lm_raw = lm_raw[:_LM_BODY]

        # Pad to at least 5 columns (x, y, z, visibility, presence)
        while lm_raw.shape[1] < 5:
            lm_raw = np.concatenate(
                [lm_raw, np.ones((33, 1), dtype=np.float32)], axis=1
            )

        # Output 1: pose present flag — raw logit or already sigmoid
        flag_raw = float(np.squeeze(outs[1])) if len(outs) > 1 else 1.0
        pose_flag = float(_sigmoid(np.array([flag_raw]))[0])
        return lm_raw[:, :5], pose_flag

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, rgb: np.ndarray) -> _PoseResult:
        """
        Run BlazePose on a single RGB uint8 frame (H, W, 3).
        Returns _PoseResult with .pose_landmarks matching MediaPipe's API.
        """
        if rgb is None or rgb.size == 0:
            return _PoseResult(None)

        det = self._detect(rgb)
        return self._process_detection(rgb, det)

    def _process_detection(self, rgb: np.ndarray, det: Optional[np.ndarray]) -> _PoseResult:
        """Run landmark inference for a pre-selected detector result."""
        h, w = rgb.shape[:2]
        if det is None:
            return _PoseResult(None)

        cx, cy, size, angle = _roi_from_detection(det, w, h)
        return self._process_roi(rgb, cx, cy, size, angle)

    def _process_roi(
        self, rgb: np.ndarray, cx: float, cy: float, size: float, angle: float
    ) -> _PoseResult:
        """Run landmark inference for a pre-selected ROI."""
        h, w = rgb.shape[:2]
        if size < 16:
            return _PoseResult(None)

        crop = _crop_roi(rgb, cx, cy, size, angle)
        lm_raw, pose_flag = self._landmark(crop)
        if pose_flag < _POSE_FLAG_THRESH or lm_raw is None:
            return _PoseResult(None)

        _, M_inv = _build_affine(cx, cy, size, angle)
        return _PoseResult(_unproject(lm_raw, M_inv, w, h))

    def estimate_batch(
        self, frames: List[np.ndarray], batch_size: int = 16
    ) -> List[_PoseResult]:
        """
        Process a list of RGB uint8 frames.

        Runs detection + landmark sequentially (GPU stays warm between calls,
        so throughput is still ~10–15× faster than MediaPipe CPU).
        `batch_size` param is reserved for future true-batch support.
        """
        results: List[_PoseResult] = []
        prev_center: Optional[np.ndarray] = None
        prev_height: Optional[float] = None
        prev_roi: Optional[Tuple[float, float, float, float]] = None
        rejected_switches = 0
        roi_tracks = 0
        roi_track_streak = 0

        for frame in frames:
            if frame is None or frame.size == 0:
                results.append(_PoseResult(None))
                continue

            candidates = self._detect_candidates(frame)
            det = _select_tracked_detection(candidates, prev_center, prev_height)
            if det is None and prev_center is not None and candidates:
                rejected_switches += 1

            if det is not None:
                res = self._process_detection(frame, det)
                roi_track_streak = 0
            elif prev_roi is not None and roi_track_streak < _MAX_ROI_TRACK_FRAMES:
                res = self._process_roi(frame, *prev_roi)
                if res.pose_landmarks is not None:
                    roi_tracks += 1
                    roi_track_streak += 1
            else:
                res = _PoseResult(None)

            if res.pose_landmarks is not None:
                h, w = frame.shape[:2]
                if det is not None:
                    prev_center, prev_height = _detection_track_state(det)
                    prev_roi = _roi_from_detection(det, w, h)
                elif prev_roi is not None:
                    next_roi = _roi_from_landmarks(res.pose_landmarks.landmark, w, h)
                    if next_roi is not None:
                        old_c = np.array([prev_roi[0] / w, prev_roi[1] / h], dtype=np.float32)
                        new_c = np.array([next_roi[0] / w, next_roi[1] / h], dtype=np.float32)
                        size_ratio = next_roi[2] / max(prev_roi[2], 1e-6)
                        if float(np.linalg.norm(new_c - old_c)) < 0.16 and 0.65 <= size_ratio <= 1.55:
                            prev_roi = next_roi
                            prev_center = new_c
            results.append(res)

        if rejected_switches:
            logger.info("Subject tracker rejected %d likely person switches", rejected_switches)
        if roi_tracks:
            logger.info("Subject tracker recovered %d frames from previous ROI", roi_tracks)
        return results

    def close(self) -> None:
        """No-op: ONNX Runtime sessions are cleaned up by the GC."""
        pass


# ---------------------------------------------------------------------------
# Provider selection helper
# ---------------------------------------------------------------------------

def _rocm_device_present() -> bool:
    """Return True only if at least one ROCm-capable GPU device is fully accessible."""
    try:
        import ctypes
        hip = ctypes.CDLL("libamdhip64.so.6")
        count = ctypes.c_int(0)
        ret = hip.hipGetDeviceCount(ctypes.byref(count))
        if ret != 0 or count.value <= 0:
            return False
        props = (ctypes.c_byte * 1024)()
        ret2 = hip.hipGetDeviceProperties(ctypes.byref(props), 0)
        return ret2 == 0
    except Exception:
        return False


def _pick_providers(ort) -> List[str]:
    """Build provider list: MIGraphX → ROCm → CPU.
    GPU EPs are only included when an actual ROCm device is present; otherwise
    we go straight to CPU to avoid a hard crash in ORT's provider init."""
    available = set(ort.get_available_providers())
    providers: List[str] = []
    if _rocm_device_present():
        for ep in ("MIGraphXExecutionProvider", "ROCMExecutionProvider"):
            if ep in available:
                providers.append(ep)
    providers.append("CPUExecutionProvider")
    if len(providers) > 1:
        logger.info("GPU device detected — using providers: %s", providers)
    else:
        logger.info("No ROCm device found — falling back to CPUExecutionProvider")
    return providers


def is_gpu_available() -> bool:
    """Quick check: returns True if MIGraphX or ROCM EP is usable."""
    try:
        import onnxruntime as ort
        avail = set(ort.get_available_providers())
        return bool(avail & {"MIGraphXExecutionProvider", "ROCMExecutionProvider"})
    except Exception:
        return False
