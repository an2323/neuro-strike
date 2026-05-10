#!/usr/bin/env python3
"""
NeuroStrike — offline football strike video post-processor.

Input: path to a video file (any format OpenCV can decode).
Output: annotated MP4 with MediaPipe pose, Savitzky–Golay smoothing, a 30-frame
“perfect strike” ghost overlay, error vectors, power meter, and form-match %.

Usage:
  python strike_video_processor.py --input kick.mp4 --output kick_ghost.mp4

Optional: pip install -r requirements_strike_video.txt (adds scipy for Savitzky–Golay).
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np

try:
    from scipy.signal import savgol_filter
except ImportError:  # pragma: no cover - optional dependency
    savgol_filter = None  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ROCm-friendly default; override with MEDIAPIPE_DISABLE_GPU=1 for CPU-only.
import os

os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "0")

GHOST_FRAMES = 30
_HIP_IDXS = (23, 24)
_SHOULDER_IDXS = (11, 12)


def _try_ffmpeg_browser_mp4(path: Path) -> None:
    """Re-encode to H.264 + yuv420p + faststart so <video> and direct URL playback work reliably."""
    if not shutil.which("ffmpeg"):
        logger.warning(
            "ffmpeg not on PATH — MP4 may use mpeg4 (mp4v) and not play in Chrome/Safari. "
            "Install: apt-get install -y ffmpeg"
        )
        return
    tmp = path.with_name(path.stem + "._h264_.mp4")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                str(tmp),
            ],
            check=True,
            capture_output=True,
            timeout=7200,
        )
        tmp.replace(path)
        logger.info("ffmpeg: H.264 + faststart remux OK (browser-friendly)")
    except Exception as exc:
        logger.warning("ffmpeg remux failed (%s); keeping OpenCV output", exc)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
# Reference ankle speed (px/s) for full power bar
POWER_METER_REF_SPEED = 900.0
# Mean normalized joint error below this maps toward 100% form match
FORM_MATCH_SCALE = 0.12
# Joints used for form match (core + kicking chain); indices are BlazePose order
FORM_MATCH_LM_INDICES = (
    0,
    11,
    12,
    13,
    14,
    15,
    16,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
)


def build_perfect_strike_ghost() -> np.ndarray:
    """
    Hardcoded 30-frame perfect strike in normalized image coordinates (x,y in [0,1]).
    Right-leg instep-style motion; camera-facing frontal template.
    Shape: (30, 33, 2).
    """
    mp_pose = mp.solutions.pose

    def base_standing() -> np.ndarray:
        p = np.zeros((33, 2), dtype=np.float64)
        # Rough BlazePose-normalized layout (frontal, feet apart).
        p[0] = (0.50, 0.14)  # NOSE
        p[2] = (0.50, 0.16)  # RIGHT_EYE
        p[5] = (0.50, 0.16)  # LEFT_EYE
        p[7] = (0.48, 0.18)  # LEFT_EAR
        p[8] = (0.52, 0.18)  # RIGHT_EAR
        p[11] = (0.40, 0.26)  # LEFT_SHOULDER
        p[12] = (0.60, 0.26)  # RIGHT_SHOULDER
        p[13] = (0.38, 0.38)  # LEFT_ELBOW
        p[14] = (0.62, 0.38)  # RIGHT_ELBOW
        p[15] = (0.36, 0.50)  # LEFT_WRIST
        p[16] = (0.64, 0.50)  # RIGHT_WRIST
        p[23] = (0.44, 0.52)  # LEFT_HIP
        p[24] = (0.56, 0.52)  # RIGHT_HIP
        p[25] = (0.44, 0.72)  # LEFT_KNEE
        p[26] = (0.56, 0.68)  # RIGHT_KNEE — start slightly flexed
        p[27] = (0.44, 0.90)  # LEFT_ANKLE
        p[28] = (0.56, 0.88)  # RIGHT_ANKLE
        p[29] = (0.44, 0.94)  # LEFT_HEEL
        p[30] = (0.56, 0.92)  # RIGHT_HEEL
        p[31] = (0.42, 0.94)  # LEFT_FOOT_INDEX
        p[32] = (0.58, 0.90)  # RIGHT_FOOT_INDEX
        # Mid spine / hips helpers
        p[9] = (0.49, 0.22)  # MOUTH_LEFT
        p[10] = (0.51, 0.22)  # MOUTH_RIGHT
        return p

    key_t = np.array([0.0, 0.25, 0.5, 0.72, 1.0], dtype=np.float64)
    keyframes: List[np.ndarray] = []

    k0 = base_standing()
    keyframes.append(k0.copy())

    k1 = k0.copy()
    # Wind-up: lift right knee, shift trunk slightly
    k1[24] = (0.57, 0.50)
    k1[26] = (0.58, 0.58)
    k1[28] = (0.62, 0.70)
    k1[30] = (0.62, 0.74)
    k1[32] = (0.64, 0.72)
    k1[12] = (0.61, 0.25)
    k1[16] = (0.66, 0.46)
    keyframes.append(k1)

    k2 = k1.copy()
    # Strike extension: leg forward-up, torso lean
    k2[0] = (0.52, 0.15)
    k2[11] = (0.39, 0.27)
    k2[12] = (0.62, 0.24)
    k2[23] = (0.43, 0.54)
    k2[24] = (0.58, 0.52)
    k2[26] = (0.55, 0.48)
    k2[28] = (0.48, 0.42)
    k2[30] = (0.47, 0.44)
    k2[32] = (0.46, 0.40)
    k2[25] = (0.44, 0.74)
    k2[27] = (0.44, 0.90)
    keyframes.append(k2)

    k3 = k2.copy()
    # Follow-through low
    k3[26] = (0.52, 0.62)
    k3[28] = (0.50, 0.78)
    k3[32] = (0.48, 0.82)
    k3[12] = (0.60, 0.26)
    keyframes.append(k3)

    k4 = base_standing()
    k4[26] = (0.56, 0.70)
    k4[28] = (0.56, 0.88)
    k4[32] = (0.58, 0.90)
    keyframes.append(k4)

    stacked = np.stack(keyframes, axis=0)  # (5, 33, 2)
    t_dst = np.linspace(0.0, 4.0, GHOST_FRAMES, dtype=np.float64)
    out = np.zeros((GHOST_FRAMES, 33, 2), dtype=np.float64)
    for lm in range(33):
        for d in range(2):
            out[:, lm, d] = np.interp(t_dst, key_t, stacked[:, lm, d])
    return np.clip(out, 0.02, 0.98)


def _savgol_1d(x: np.ndarray) -> np.ndarray:
    """Savitzky–Golay along time for a single coordinate series."""
    n = int(x.shape[0])
    if n < 5 or savgol_filter is None:
        return np.asarray(x, dtype=np.float64).copy()
    window = min(21, n if n % 2 == 1 else n - 1)
    window = max(5, window if window % 2 == 1 else window - 1)
    if window > n or window < 5:
        return np.asarray(x, dtype=np.float64).copy()
    polyorder = 3
    if polyorder >= window:
        polyorder = max(2, window - 2)
    try:
        return savgol_filter(np.asarray(x, dtype=np.float64), window_length=window, polyorder=polyorder, mode="interp")
    except Exception as e:  # pragma: no cover
        logger.warning("Savitzky–Golay failed (%s); using raw series.", e)
        return np.asarray(x, dtype=np.float64).copy()


def _forward_fill_landmarks(
    seq: np.ndarray,
    vis: np.ndarray,
    min_vis: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """seq (T,33,2), vis (T,33). Fill missing frames with last good pose."""
    out = seq.copy()
    out_v = vis.copy()
    last_good: Optional[np.ndarray] = None
    for t in range(seq.shape[0]):
        if np.nanmax(out_v[t]) < min_vis and last_good is not None:
            out[t] = last_good
            out_v[t] = np.maximum(out_v[t], 0.35)
        elif np.nanmax(out_v[t]) >= min_vis:
            last_good = out[t].copy()
    return out, out_v


def _ankle_speeds(
    seq: np.ndarray,
    vis: np.ndarray,
    fps: float,
) -> np.ndarray:
    """Per-frame max(L,R) ankle speed in normalized coords / second (scale × ref later)."""
    t = seq.shape[0]
    sp = np.zeros(t, dtype=np.float64)
    la, ra = 27, 28
    for i in range(1, t):
        dt = 1.0 / max(fps, 1e-3)
        if vis[i, la] > 0.2 and vis[i - 1, la] > 0.2:
            dl = np.linalg.norm(seq[i, la] - seq[i - 1, la]) / dt
        else:
            dl = 0.0
        if vis[i, ra] > 0.2 and vis[i - 1, ra] > 0.2:
            dr = np.linalg.norm(seq[i, ra] - seq[i - 1, ra]) / dt
        else:
            dr = 0.0
        sp[i] = max(dl, dr)
    return sp


def _find_peak_frame(speeds: np.ndarray) -> int:
    i = int(np.argmax(speeds))
    if speeds[i] < 1e-6:
        return len(speeds) // 2
    return i


def _strike_phase_window(speeds: np.ndarray, peak_idx: int) -> Tuple[int, int]:
    """Estimate wind-up -> follow-through window around peak speed."""
    n = int(speeds.shape[0])
    if n <= 1:
        return (0, max(0, n - 1))
    peak = float(speeds[peak_idx]) if 0 <= peak_idx < n else 0.0
    if peak <= 1e-6:
        span = min(max(1, GHOST_FRAMES), n)
        s = max(0, (n - span) // 2)
        return (s, min(n - 1, s + span - 1))
    thr = max(peak * 0.18, float(np.percentile(speeds, 70)))
    start = int(peak_idx)
    end = int(peak_idx)
    while start > 0 and float(speeds[start - 1]) >= thr:
        start -= 1
    while end < n - 1 and float(speeds[end + 1]) >= thr:
        end += 1
    # Provide context around detected motion.
    start = max(0, start - 8)
    end = min(n - 1, end + 10)
    return (start, end)


def _ghost_index_for_frame(frame_idx: int, phase_start: int, phase_end: int) -> Optional[int]:
    """Loop ghost frames through the whole detected strike phase."""
    if frame_idx < phase_start or frame_idx > phase_end:
        return None
    return int((frame_idx - phase_start) % GHOST_FRAMES)


def _midpoint_xy(points_xy: np.ndarray, idxs: Tuple[int, int], vis: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    vals: List[np.ndarray] = []
    for i in idxs:
        if i < 0 or i >= points_xy.shape[0]:
            continue
        p = points_xy[i]
        if not np.all(np.isfinite(p)):
            continue
        if vis is not None and (i >= vis.shape[0] or float(vis[i]) < 0.2):
            continue
        vals.append(np.asarray(p, dtype=np.float64))
    if not vals:
        return None
    return np.mean(np.stack(vals, axis=0), axis=0)


def _transform_ghost_to_user(ghost_xy_norm: np.ndarray, user_xy_norm: np.ndarray, user_vis: np.ndarray) -> np.ndarray:
    """
    Scale ghost by user torso (shoulder↔hip) and center on user's hip midpoint.
    Works in normalized coords.
    """
    g_hip = _midpoint_xy(ghost_xy_norm, _HIP_IDXS, None)
    g_sh = _midpoint_xy(ghost_xy_norm, _SHOULDER_IDXS, None)
    u_hip = _midpoint_xy(user_xy_norm, _HIP_IDXS, user_vis)
    u_sh = _midpoint_xy(user_xy_norm, _SHOULDER_IDXS, user_vis)
    if g_hip is None or u_hip is None:
        return ghost_xy_norm.copy()
    scale = 1.0
    if g_sh is not None and u_sh is not None:
        g_torso = float(np.linalg.norm(g_sh - g_hip))
        u_torso = float(np.linalg.norm(u_sh - u_hip))
        if g_torso > 1e-4 and np.isfinite(u_torso):
            scale = float(np.clip(u_torso / g_torso, 0.35, 2.8))
    return (ghost_xy_norm - g_hip) * scale + u_hip


def _is_cornerish_norm(pt_norm: np.ndarray, eps: float = 0.03) -> bool:
    """Skip vectors around (0,0) / (1,1) to avoid corner-shooting artifacts."""
    if not np.all(np.isfinite(pt_norm)):
        return True
    x, y = float(pt_norm[0]), float(pt_norm[1])
    return (x <= eps and y <= eps) or (x >= 1.0 - eps and y >= 1.0 - eps)


def _form_match_percent(
    user_xy_norm: np.ndarray,
    ghost_xy_norm: np.ndarray,
    vis: np.ndarray,
) -> float:
    errs: List[float] = []
    for lm in FORM_MATCH_LM_INDICES:
        if vis[lm] < 0.2:
            continue
        uu, gg = user_xy_norm[lm], ghost_xy_norm[lm]
        if not (np.all(np.isfinite(uu)) and np.all(np.isfinite(gg))):
            continue
        du = np.linalg.norm(uu - gg)
        errs.append(float(du))
    if not errs:
        return 0.0
    mean_e = float(np.mean(errs))
    # 0 error -> 100%, mean_e >= scale -> ~0%
    score = 100.0 * max(0.0, 1.0 - mean_e / FORM_MATCH_SCALE)
    return float(np.clip(score, 0.0, 100.0))


def _draw_dashed_line(
    img: np.ndarray,
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash_len: int = 8,
    gap: int = 6,
) -> None:
    d = np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=np.float64)
    dist = float(np.linalg.norm(d))
    if dist < 1e-6:
        return
    d /= dist
    drawn = 0.0
    toggle = True
    while drawn < dist:
        step = dash_len if toggle else gap
        t0 = drawn
        t1 = min(dist, drawn + step)
        if toggle:
            a = (int(round(p1[0] + d[0] * t0)), int(round(p1[1] + d[1] * t0)))
            b = (int(round(p1[0] + d[0] * t1)), int(round(p1[1] + d[1] * t1)))
            cv2.line(img, a, b, color, thickness, cv2.LINE_AA)
        drawn = t1
        toggle = not toggle


def _finite_pt(pts_px: np.ndarray, i: int) -> bool:
    if i < 0 or i >= pts_px.shape[0]:
        return False
    x, y = float(pts_px[i, 0]), float(pts_px[i, 1])
    return np.isfinite(x) and np.isfinite(y)


def _draw_skeleton_lines(
    img: np.ndarray,
    pts_px: np.ndarray,
    connections: Sequence[Tuple[int, int]],
    color: Tuple[int, int, int],
    thickness: int = 3,
) -> None:
    for a, b in connections:
        ia, ib = int(a), int(b)
        if ia >= pts_px.shape[0] or ib >= pts_px.shape[0]:
            continue
        if not _finite_pt(pts_px, ia) or not _finite_pt(pts_px, ib):
            continue
        pa = (int(round(pts_px[ia, 0])), int(round(pts_px[ia, 1])))
        pb = (int(round(pts_px[ib, 0])), int(round(pts_px[ib, 1])))
        # Skip segments where template / detection left landmarks unset (near origin).
        if abs(pa[0]) + abs(pa[1]) < 3 or abs(pb[0]) + abs(pb[1]) < 3:
            continue
        cv2.line(img, pa, pb, color, thickness, cv2.LINE_AA)


def _draw_skeleton_points(img: np.ndarray, pts_px: np.ndarray, color: Tuple[int, int, int], r: int = 4) -> None:
    for i in range(pts_px.shape[0]):
        if not _finite_pt(pts_px, i):
            continue
        x, y = int(round(pts_px[i, 0])), int(round(pts_px[i, 1]))
        if abs(x) + abs(y) < 3:
            continue
        cv2.circle(img, (x, y), r, color, -1, cv2.LINE_AA)


def _blend_color_layer(
    base: np.ndarray,
    layer: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Blend non-black pixels from layer onto base (semi-transparent effect)."""
    mask = (layer.max(axis=2) > 0).astype(np.float32)
    a = alpha * mask
    a3 = np.stack([a, a, a], axis=2)
    out = np.clip(base.astype(np.float32) * (1.0 - a3) + layer.astype(np.float32) * a3, 0, 255).astype(np.uint8)
    return out


def _draw_power_meter(
    img: np.ndarray,
    fill_ratio: float,
    margin: int = 12,
    bar_w: int = 22,
) -> None:
    h, w = img.shape[:2]
    fill_ratio = float(np.clip(fill_ratio, 0.0, 1.0))
    x0 = margin
    y0 = int(h * 0.15)
    y1 = int(h * 0.85)
    inner_h = y1 - y0 - 4
    fill_h = int(inner_h * fill_ratio)
    # Shell
    cv2.rectangle(img, (x0, y0), (x0 + bar_w, y1), (40, 40, 40), -1)
    cv2.rectangle(img, (x0, y0), (x0 + bar_w, y1), (200, 200, 200), 2)
    # Fill (neon cyan -> hot)
    y_fill_top = y1 - 2 - fill_h
    for dy in range(fill_h):
        t = dy / max(inner_h, 1)
        b = int(255 * (1 - t * 0.5))
        g = int(180 + 75 * t)
        r = int(40 + 200 * t)
        cv2.line(
            img,
            (x0 + 2, y_fill_top + dy),
            (x0 + bar_w - 2, y_fill_top + dy),
            (b, g, r),
            1,
        )
    cv2.putText(
        img,
        "POWER",
        (x0 - 2, y0 - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )


def process_video(
    input_path: Path,
    output_path: Path,
    min_detection: float = 0.5,
    min_tracking: float = 0.5,
) -> None:
    mp_pose = mp.solutions.pose
    ghost_norm = build_perfect_strike_ghost()
    connections = [(int(a), int(b)) for a, b in mp_pose.POSE_CONNECTIONS]

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info("Input %s  %dx%d  %.2f fps  ~%d frames", input_path.name, w, h, fps, n_frames)

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=False,  # we apply Savitzky–Golay offline
        min_detection_confidence=min_detection,
        min_tracking_confidence=min_tracking,
    )

    raw_seq: List[np.ndarray] = []
    raw_vis: List[np.ndarray] = []
    frames_read = 0

    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        frames_read += 1
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        xy = np.full((33, 2), np.nan, dtype=np.float64)
        v = np.zeros(33, dtype=np.float64)
        if res.pose_landmarks:
            for idx, lm in enumerate(res.pose_landmarks.landmark):
                if idx >= 33:
                    break
                xy[idx, 0] = float(lm.x)
                xy[idx, 1] = float(lm.y)
                v[idx] = float(lm.visibility)
        raw_seq.append(xy)
        raw_vis.append(v)

    cap.release()
    pose.close()

    if frames_read == 0:
        raise RuntimeError("No frames read from video.")

    seq = np.stack(raw_seq, axis=0)  # (T, 33, 2)
    vis = np.stack(raw_vis, axis=0)  # (T, 33)
    seq, vis = _forward_fill_landmarks(seq, vis)

    if savgol_filter is None:
        logger.warning("scipy not installed; install scipy for Savitzky–Golay smoothing.")

    # Savitzky–Golay on x and y independently per landmark
    smooth = seq.copy()
    for lm in range(33):
        for d in range(2):
            smooth[:, lm, d] = _savgol_1d(seq[:, lm, d])
    # S-G can yield NaN at edges or where input had gaps; fall back to raw seq then center.
    bad = ~np.isfinite(smooth)
    if np.any(bad):
        smooth = np.where(np.isfinite(smooth), smooth, seq)
        bad2 = ~np.isfinite(smooth)
        if np.any(bad2):
            smooth[bad2] = 0.5

    speeds_norm = _ankle_speeds(smooth, vis, fps)
    peak_idx = _find_peak_frame(speeds_norm)
    phase_start, phase_end = _strike_phase_window(speeds_norm, peak_idx)
    logger.info(
        "Strike phase: start=%d peak=%d end=%d / %d",
        phase_start,
        peak_idx,
        phase_end,
        frames_read,
    )

    # --- Second pass: encode ---
    cap = cv2.VideoCapture(str(input_path))
    # Prefer H.264-style fourcc for OpenCV when it works; mp4v (mpeg4) often will not play in browsers.
    writer: Optional[cv2.VideoWriter] = None
    for codec in ("avc1", "H264", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        wri = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        if wri.isOpened():
            writer = wri
            logger.info("VideoWriter using fourcc=%s", codec)
            break
        logger.warning("VideoWriter fourcc=%s failed; trying next codec", codec)
    if writer is None:
        raise RuntimeError("Could not open VideoWriter with mp4v or avc1.")

    fi = 0
    neon_blue = (255, 128, 0)  # BGR vivid blue
    gold = (0, 200, 255)  # BGR gold
    error_color = (60, 60, 255)  # red-ish

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fi >= smooth.shape[0]:
            break

        out = frame.copy()
        u_norm = smooth[fi]
        v_row = vis[fi]
        pts_user = np.zeros((33, 2), dtype=np.float64)
        for lm in range(33):
            pts_user[lm, 0] = u_norm[lm, 0] * w
            pts_user[lm, 1] = u_norm[lm, 1] * h

        # User skeleton (neon blue)
        _draw_skeleton_lines(out, pts_user, connections, neon_blue, 3)
        _draw_skeleton_points(out, pts_user, neon_blue, 4)

        ghost_g = _ghost_index_for_frame(fi, phase_start, phase_end)
        form_pct = 0.0
        if ghost_g is not None:
            g_norm = _transform_ghost_to_user(ghost_norm[ghost_g], u_norm, v_row)
            pts_ghost = np.zeros((33, 2), dtype=np.float64)
            for lm in range(33):
                pts_ghost[lm, 0] = g_norm[lm, 0] * w
                pts_ghost[lm, 1] = g_norm[lm, 1] * h

            ghost_layer = np.zeros_like(out)
            _draw_skeleton_lines(ghost_layer, pts_ghost, connections, gold, 3)
            _draw_skeleton_points(ghost_layer, pts_ghost, gold, 4)
            out = _blend_color_layer(out, ghost_layer, 0.55)

            # Error vectors (dashed red)
            for lm in FORM_MATCH_LM_INDICES:
                if v_row[lm] < 0.2:
                    continue
                if not _finite_pt(pts_user, lm) or not _finite_pt(pts_ghost, lm):
                    continue
                if _is_cornerish_norm(u_norm[lm]) or _is_cornerish_norm(g_norm[lm]):
                    continue
                p_u = (int(round(pts_user[lm, 0])), int(round(pts_user[lm, 1])))
                p_g = (int(round(pts_ghost[lm, 0])), int(round(pts_ghost[lm, 1])))
                _draw_dashed_line(out, p_u, p_g, error_color, 1, 6, 5)

            form_pct = _form_match_percent(u_norm, g_norm, v_row)

        # Power meter from normalized speed → px/s equivalent for display
        speed_px_equiv = speeds_norm[fi] * float(np.hypot(w, h))
        fill = min(1.0, speed_px_equiv / POWER_METER_REF_SPEED)
        _draw_power_meter(out, fill)

        # Form match text
        label = f"Form Match: {form_pct:.1f}%"
        if ghost_g is None:
            label = "Form Match: -- (outside strike phase)"
        tw, th = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)[0]
        bx0, bx1 = 8, 8 + tw + 16
        by0, by1 = h - th - 28, h - 8
        cv2.rectangle(out, (bx0, by0), (bx1, by1), (20, 20, 20), -1)
        cv2.rectangle(out, (bx0, by0), (bx1, by1), (0, 220, 255), 2)
        cv2.putText(
            out,
            label,
            (16, h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (240, 250, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(out)
        fi += 1

    cap.release()
    writer.release()
    logger.info("Wrote %s (%d frames)", output_path, fi)
    _try_ffmpeg_browser_mp4(output_path)


def main() -> None:
    p = argparse.ArgumentParser(description="Football strike video processor with ghost overlay.")
    p.add_argument("--input", "-i", type=Path, required=True, help="Input video path")
    p.add_argument("--output", "-o", type=Path, required=True, help="Output MP4 path")
    p.add_argument("--min-detection", type=float, default=0.5)
    p.add_argument("--min-tracking", type=float, default=0.5)
    args = p.parse_args()

    if not args.input.is_file():
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        process_video(args.input, args.output, args.min_detection, args.min_tracking)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
