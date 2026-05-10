#!/usr/bin/env python3
"""
NeuroStrike — offline football strike video post-processor.

Input: path to a video file (any format OpenCV can decode).
Output: annotated MP4 with MediaPipe pose, Savitzky–Golay smoothing, and a
single-skeleton biomechanical heatmap (blue->red error gradient), plus power meter and form-match %.

Usage:
  python strike_video_processor.py --input kick.mp4 --output kick_heatmap.mp4

Optional: pip install -r requirements_strike_video.txt (adds scipy for Savitzky–Golay).
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
JOINT_ERROR_MAX_DEG = 30.0
SUBJECT_MAX_CENTROID_JUMP = 0.16
SUBJECT_MAX_HEIGHT_RATIO = 1.45
SUBJECT_MIN_HEIGHT_RATIO = 0.62
STORYBOARD_PHASE_TARGETS: Tuple[Tuple[str, float], ...] = (
    ("approach", 0.10),
    ("windup", 0.30),
    ("impact", 0.50),
    ("follow", 0.80),
)
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

MAIN_JOINT_TRIPLETS: Dict[int, Tuple[int, int, int]] = {
    # shoulder: elbow-shoulder-hip
    11: (13, 11, 23),
    12: (14, 12, 24),
    # hip: shoulder-hip-knee
    23: (11, 23, 25),
    24: (12, 24, 26),
    # knee: hip-knee-ankle
    25: (23, 25, 27),
    26: (24, 26, 28),
    # ankle: knee-ankle-foot
    27: (25, 27, 31),
    28: (26, 28, 32),
}

HEATMAP_BONES: Tuple[Tuple[int, int], ...] = (
    (11, 12),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (27, 31),
    (24, 26),
    (26, 28),
    (28, 32),
)



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


def _pose_centroid_height(xy: np.ndarray, vis_row: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    """Return subject centroid and approximate body height in normalized coords."""
    idxs = (11, 12, 23, 24, 25, 26, 27, 28)
    pts: List[np.ndarray] = []
    for i in idxs:
        if i >= xy.shape[0]:
            continue
        if not np.all(np.isfinite(xy[i])):
            continue
        if i < vis_row.shape[0] and float(vis_row[i]) < 0.2:
            continue
        pts.append(xy[i])
    if len(pts) < 3:
        return None, 0.0
    arr = np.stack(pts, axis=0)
    return np.mean(arr, axis=0), float(np.max(arr[:, 1]) - np.min(arr[:, 1]))


def _stabilize_subject_track(seq: np.ndarray, vis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Suppress sudden jumps to another person in the frame.

    This keeps cinematic zoom and heatmap tied to one athlete.
    """
    out = seq.copy()
    out_v = vis.copy()
    prev_c: Optional[np.ndarray] = None
    prev_h: Optional[float] = None
    for t in range(out.shape[0]):
        c, h = _pose_centroid_height(out[t], out_v[t])
        if c is None or h <= 1e-6:
            if t > 0:
                out[t] = out[t - 1]
                out_v[t] = out_v[t - 1] * 0.98
            continue
        if prev_c is not None and prev_h is not None and prev_h > 1e-6:
            jump = float(np.linalg.norm(c - prev_c))
            ratio = h / prev_h
            switched = (
                jump > SUBJECT_MAX_CENTROID_JUMP
                and (ratio > SUBJECT_MAX_HEIGHT_RATIO or ratio < SUBJECT_MIN_HEIGHT_RATIO)
            )
            if switched and t > 0:
                out[t] = out[t - 1]
                out_v[t] = out_v[t - 1] * 0.985
                c, h = _pose_centroid_height(out[t], out_v[t])
        prev_c = c
        prev_h = h
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
    """
    Find the impact-like peak robustly.

    We intentionally ignore very early/late video regions where setup movement
    and stop-motion noise often produce false maxima.
    """
    n = int(speeds.shape[0])
    if n <= 2:
        return max(0, n // 2)

    # Light temporal smoothing to suppress one-frame spikes.
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
    kernel /= float(np.sum(kernel))
    s = np.convolve(speeds.astype(np.float64), kernel, mode="same")

    # Action usually happens away from intro/outro; constrain search window.
    lo = int(max(0, n * 0.15))
    hi = int(min(n, max(lo + 1, n * 0.95)))
    window = s[lo:hi]
    if window.size == 0:
        window = s
        lo = 0

    i = lo + int(np.argmax(window))
    peak = float(s[i])
    if peak < 1e-6:
        return n // 2
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
    start = max(0, start - 10)
    end = min(n - 1, end + 14)

    # Prevent extremely short windows from noisy thresholds.
    min_span = min(max(18, GHOST_FRAMES // 2), n - 1) if n > 1 else 0
    curr_span = end - start
    if curr_span < min_span:
        pad = (min_span - curr_span + 1) // 2
        start = max(0, start - pad)
        end = min(n - 1, end + pad)
    return (start, end)


def _in_strike_phase(frame_idx: int, phase_start: int, phase_end: int) -> bool:
    """Return True if frame_idx is within the detected strike phase."""
    return phase_start <= frame_idx <= phase_end


def _phase_t(frame_idx: int, phase_start: int, phase_end: int) -> float:
    """Normalised position within the strike phase: 0.0 wind-up, 0.5 impact, 1.0 follow-through."""
    span = max(1, phase_end - phase_start)
    return float(np.clip((frame_idx - phase_start) / span, 0.0, 1.0))


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



def _is_cornerish_norm(pt_norm: np.ndarray, eps: float = 0.03) -> bool:
    """Skip vectors around (0,0) / (1,1) to avoid corner-shooting artifacts."""
    if not np.all(np.isfinite(pt_norm)):
        return True
    x, y = float(pt_norm[0]), float(pt_norm[1])
    return (x <= eps and y <= eps) or (x >= 1.0 - eps and y >= 1.0 - eps)


# ─── BlazePose leg landmark indices ─────────────────────────────────────────
_LEFT_LEG  = {"hip": 23, "knee": 25, "ankle": 27, "heel": 29, "foot": 31}
_RIGHT_LEG = {"hip": 24, "knee": 26, "ankle": 28, "heel": 30, "foot": 32}

# ─── Biomechanical correction targets ────────────────────────────────────────
_LEAN_BAND_LO_DEG = 5.0
_LEAN_BAND_HI_DEG = 10.0
_LEAN_TARGET_DEG = 7.5  # within band above (degrees forward from vertical in image coords)
_IMPACT_INTERIOR_DEG = 172.0  # hip-knee-ankle interior angle at ball contact (almost straight)
# Wind-up: extra knee flex vs a neutral instep template (see _ideal_kicking_knee_interior_deg)
_WINDUP_EXTRA_DEG = 20.0


def _set_joint_angle(
    proximal: np.ndarray,
    middle: np.ndarray,
    distal: np.ndarray,
    target_deg: float,
) -> np.ndarray:
    """
    Return a new distal position so the angle at middle equals target_deg.
    The distal segment length is preserved; the rotation direction maintains
    the current anatomical orientation (same side as original distal).
    """
    v1 = proximal - middle   # bone toward proximal joint
    v2 = distal - middle     # bone toward distal joint
    l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if l1 < 1e-6 or l2 < 1e-6:
        return distal.copy()
    v1u = v1 / l1
    target_rad = np.radians(np.clip(target_deg, 1.0, 179.0))
    # Choose a perpendicular to v1u that lies on the same side as v2
    perp = np.array([-v1u[1], v1u[0]])   # 90° CCW from v1u
    if np.dot(v2, perp) < 0:
        perp = -perp                       # flip to match the anatomical side
    # Construct the new distal direction at exactly target_rad from v1u
    new_v2_unit = np.cos(target_rad) * v1u + np.sin(target_rad) * perp
    return middle + l2 * new_v2_unit


def _snap_leg_interior(
    g: np.ndarray,
    hip_i: int,
    knee_i: int,
    ankle_i: int,
    heel_i: int,
    foot_i: int,
    target_interior_deg: float,
) -> None:
    """Force hip-knee-ankle interior angle; move ankle/foot/heel by translation (bone lengths preserved)."""
    if (
        hip_i >= g.shape[0]
        or knee_i >= g.shape[0]
        or ankle_i >= g.shape[0]
        or not np.all(np.isfinite(g[hip_i]))
        or not np.all(np.isfinite(g[knee_i]))
        or not np.all(np.isfinite(g[ankle_i]))
    ):
        return
    new_ankle = _set_joint_angle(g[hip_i], g[knee_i], g[ankle_i], float(target_interior_deg))
    shift = new_ankle - g[ankle_i]
    g[ankle_i] += shift
    for idx in (heel_i, foot_i):
        if idx < g.shape[0] and np.all(np.isfinite(g[idx])):
            g[idx] += shift


def _ideal_base_knee_interior_deg(phase_t: float) -> float:
    """
    Template instep kinematic (interior hip-knee-ankle angle vs strike phase).

    Sharp wind-up tucked knee near mid-wind-up, near-full extension centered on impact,
    softer follow-through. Values chosen so the silhouette reads as textbook form.
    """
    t_breaks = np.array([0.0, 0.16, 0.28, 0.36, 0.44, 0.52, 0.62, 0.82, 1.0], dtype=np.float64)
    ideals = np.array([118.0, 98.0, 78.0, 64.0, 108.0, 176.0, 170.0, 136.0, 124.0], dtype=np.float64)
    return float(np.interp(float(np.clip(phase_t, 0.0, 1.0)), t_breaks, ideals))


def _ideal_kicking_knee_interior_deg(phase_t: float) -> float:
    """
    User rule: backswing shows ~20° extra knee flex vs neutral template (more power pocket).
    """
    base = _ideal_base_knee_interior_deg(phase_t)
    wind = float(np.clip(1.0 - phase_t / 0.38, 0.0, 1.0))
    # Smaller interior angle = deeper knee bend
    return float(max(28.0, base - _WINDUP_EXTRA_DEG * wind))


def _ideal_support_knee_interior_deg(phase_t: float) -> float:
    """Plant leg: stable, slightly flexed — contrast with kicking leg."""
    t_breaks = np.array([0.0, 0.35, 0.50, 0.68, 1.0], dtype=np.float64)
    ideals = np.array([158.0, 155.0, 166.0, 162.0, 160.0], dtype=np.float64)
    return float(np.interp(float(np.clip(phase_t, 0.0, 1.0)), t_breaks, ideals))


def _detect_kicking_leg(smooth: np.ndarray, vis: np.ndarray, peak_idx: int) -> str:
    """Identify the kicking leg by comparing ankle displacement near the peak frame."""
    lo = max(0, peak_idx - 4)
    hi = min(smooth.shape[0], peak_idx + 5)
    sl = sr = 0.0
    for i in range(lo + 1, hi):
        if vis[i, 27] > 0.2 and vis[i - 1, 27] > 0.2:
            sl += float(np.linalg.norm(smooth[i, 27] - smooth[i - 1, 27]))
        if vis[i, 28] > 0.2 and vis[i - 1, 28] > 0.2:
            sr += float(np.linalg.norm(smooth[i, 28] - smooth[i - 1, 28]))
    kicking = "LEFT" if sl >= sr else "RIGHT"
    logger.info("Kicking leg detected: %s (left_disp=%.4f  right_disp=%.4f)", kicking, sl, sr)
    return kicking


def _generate_corrected_ghost(
    user_xy: np.ndarray,
    user_vis: np.ndarray,
    phase_t: float,
    kicking_side: str,
) -> np.ndarray:
    """
    Build a per-frame ideal instep coach pose anchored on the athlete.

    Hips / bone lengths follow the user in normalised image space; articulation
    is driven by phase so the golden skeleton shows textbook wind-up, extension,
    and follow-through instead of mirroring noisy input.

      Rule 1 – Lean:  torso forward tilt clamped into the coach band (~5°–10°).
      Rule 2 – Wind-up: +20° extra knee fold vs the neutral kinematic curve.
      Rule 3 – Impact: kicking leg snaps toward nearly straight (~172° interior).
      Support leg is held in a stable, slightly flexed plant shape for contrast.

    ``user_vis`` is reserved for future gating but does not limit ghost geometry.
    """
    _ = user_vis  # symmetry with callers; limb snaps use finite landmarks only

    g = user_xy.copy()
    kick = _RIGHT_LEG if kicking_side == "RIGHT" else _LEFT_LEG
    plant = _LEFT_LEG if kicking_side == "RIGHT" else _RIGHT_LEG

    # ── Rule 1: Torso forward lean (fixed coach band), not athlete-specific lean ─
    if np.all(np.isfinite(g[11])) and np.all(np.isfinite(g[12])):
        mid_hip = (g[23] + g[24]) * 0.5
        mid_sh = (g[11] + g[12]) * 0.5
        torso = mid_sh - mid_hip
        t_len = float(np.linalg.norm(torso))
        if t_len > 1e-4:
            curr_lean = float(np.degrees(np.arctan2(torso[0], -torso[1])))
            if abs(curr_lean) < 2.0:
                fwd = 1.0 if kicking_side == "RIGHT" else -1.0
            else:
                fwd = float(np.sign(curr_lean)) if abs(curr_lean) > 0.5 else 1.0
            lean_mag = float(np.clip(_LEAN_TARGET_DEG, _LEAN_BAND_LO_DEG, _LEAN_BAND_HI_DEG))
            tgt_rad = np.radians(fwd * lean_mag)
            new_torso = t_len * np.array([np.sin(tgt_rad), -np.cos(tgt_rad)])
            sh_shift = new_torso - torso
            for idx in range(17):  # head + arms rigid with shoulders
                if np.all(np.isfinite(g[idx])):
                    g[idx] = g[idx] + sh_shift

    pt = float(np.clip(phase_t, 0.0, 1.0))

    # ── Rule 2 + 3: Phase-native kicking leg kinematics (+ wind-up sharpening) ─
    kick_ideal = _ideal_kicking_knee_interior_deg(pt)
    impact_gate = float(np.clip(1.0 - abs(pt - 0.5) / 0.12, 0.0, 1.0))
    kick_ideal = kick_ideal * (1.0 - impact_gate) + _IMPACT_INTERIOR_DEG * impact_gate

    _snap_leg_interior(
        g,
        kick["hip"],
        kick["knee"],
        kick["ankle"],
        kick["heel"],
        kick["foot"],
        kick_ideal,
    )

    _snap_leg_interior(
        g,
        plant["hip"],
        plant["knee"],
        plant["ankle"],
        plant["heel"],
        plant["foot"],
        _ideal_support_knee_interior_deg(pt),
    )

    return g


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


def _joint_angle_deg(pts: np.ndarray, a: int, b: int, c: int) -> Optional[float]:
    if not (_finite_pt(pts, a) and _finite_pt(pts, b) and _finite_pt(pts, c)):
        return None
    ba = pts[a] - pts[b]
    bc = pts[c] - pts[b]
    la = float(np.linalg.norm(ba))
    lc = float(np.linalg.norm(bc))
    if la < 1e-6 or lc < 1e-6:
        return None
    ang = float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / (la * lc), -1.0, 1.0))))
    return ang


def _angle_abs_diff_deg(a1: float, a2: float) -> float:
    d = abs(float(a1) - float(a2)) % 360.0
    return min(d, 360.0 - d)


def _error_deg_to_bgr(err_deg: float) -> Tuple[int, int, int]:
    # 0 deg -> blue, JOINT_ERROR_MAX_DEG -> red.
    t = float(np.clip(err_deg / JOINT_ERROR_MAX_DEG, 0.0, 1.0))
    hue = (1.0 - t) * 240.0  # blue(240) -> red(0)
    h_cv = int(np.clip(hue / 2.0, 0, 179))
    hls = np.uint8([[[h_cv, 140, 255]]])  # bright, high saturation
    bgr = cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _draw_gradient_bone(
    img: np.ndarray,
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    c1: Tuple[int, int, int],
    c2: Tuple[int, int, int],
    thickness: int = 4,
) -> None:
    d = np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=np.float64)
    dist = float(np.linalg.norm(d))
    if dist < 1e-6:
        return
    segments = max(8, int(dist / 10.0))
    for i in range(segments):
        t0 = i / segments
        t1 = (i + 1) / segments
        q0 = (int(round(p1[0] + d[0] * t0)), int(round(p1[1] + d[1] * t0)))
        q1 = (int(round(p1[0] + d[0] * t1)), int(round(p1[1] + d[1] * t1)))
        tc = (t0 + t1) * 0.5
        col = (
            int(round(c1[0] * (1.0 - tc) + c2[0] * tc)),
            int(round(c1[1] * (1.0 - tc) + c2[1] * tc)),
            int(round(c1[2] * (1.0 - tc) + c2[2] * tc)),
        )
        cv2.line(img, q0, q1, col, thickness, cv2.LINE_AA)


def _joint_error_map_deg(user_pts: np.ndarray, ideal_pts: np.ndarray) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for j, (a, b, c) in MAIN_JOINT_TRIPLETS.items():
        ua = _joint_angle_deg(user_pts, a, b, c)
        ia = _joint_angle_deg(ideal_pts, a, b, c)
        if ua is None or ia is None:
            continue
        out[j] = _angle_abs_diff_deg(ua, ia)
    return out


def _form_match_from_joint_errors(joint_errors: Dict[int, float]) -> float:
    if not joint_errors:
        return 0.0
    vals = [float(np.clip(v, 0.0, JOINT_ERROR_MAX_DEG)) for v in joint_errors.values()]
    mean_v = float(np.mean(vals))
    return float(np.clip(100.0 * (1.0 - mean_v / JOINT_ERROR_MAX_DEG), 0.0, 100.0))


def generate_coach_verdict(metrics: Dict[str, float]) -> Dict[str, Any]:
    """
    Build structured coaching verdict from analyzed metrics.

    Returns dict with strengths, weaknesses, actionable_advice, and coaching_audio_text.
    """
    strengths: List[str] = []
    weaknesses: List[str] = []

    impact_speed = float(metrics.get("impact_speed", 0.0))
    backswing_angle = float(metrics.get("max_backswing_angle", 0.0))
    torso_stability = float(metrics.get("torso_stability", 0.0))
    overall_form = float(metrics.get("overall_form_score", 0.0))
    knee_extension = float(metrics.get("knee_extension_at_impact", 0.0))
    lean_err = float(metrics.get("lean_error", 999.0))

    if impact_speed >= 700.0:
        strengths.append("High swing velocity")
    if 70.0 <= backswing_angle <= 120.0:
        strengths.append("Efficient backswing loading")
    if torso_stability <= 7.0:
        strengths.append("Stable torso control")
    if overall_form >= 75.0:
        strengths.append("Strong overall form consistency")
    strengths = strengths[:2] if strengths else ["Good training intent and timing"]

    biggest_issue = "Inconsistent knee extension"
    if knee_extension < 150.0:
        biggest_issue = "Inconsistent knee extension"
    elif lean_err > 8.0:
        biggest_issue = "Torso lean not controlled"
    elif torso_stability > 11.0:
        biggest_issue = "Torso stability drift"
    elif overall_form < 60.0:
        biggest_issue = "General form consistency is low"
    weaknesses.append(biggest_issue)

    if "knee" in biggest_issue.lower():
        advice = "Drive your kicking knee through the ball and finish with a longer extension."
    elif "lean" in biggest_issue.lower():
        advice = "Lean your torso slightly forward through contact to keep the strike controlled and low."
    elif "stability" in biggest_issue.lower():
        advice = "Keep your core braced and chest quiet during the final swing to stabilize the strike path."
    else:
        advice = "Slow down your approach by one step and focus on matching the target body angles."

    recommended_drills: List[Dict[str, str]] = []
    if knee_extension < 155.0:
        recommended_drills.append(
            {
                "title": "Hamstring Dynamic Stretches",
                "instruction": "3 sets of 10 reps per leg to improve flexibility before strike practice.",
                "icon": "🦵",
            }
        )
    if torso_stability > 9.0:
        recommended_drills.append(
            {
                "title": "Core Stability - Planks",
                "instruction": "3 rounds of 45 seconds, keep hips level and core braced.",
                "icon": "🧱",
            }
        )
    if not recommended_drills:
        recommended_drills.append(
            {
                "title": "Controlled Ball Strikes",
                "instruction": "2 sets of 8 technical strikes focusing on repeatable body alignment.",
                "icon": "⚽",
            }
        )

    coaching_audio_text = (
        f"Great work. Strengths: {', '.join(strengths)}. "
        f"Main area to improve: {weaknesses[0]}. "
        f"Coaching tip: {advice}"
    )
    return {
        "strengths": strengths,
        "weaknesses": weaknesses,
        "actionable_advice": advice,
        "recommended_drills": recommended_drills[:2],
        "coaching_audio_text": coaching_audio_text,
    }


def _draw_heatmap_skeleton(
    img: np.ndarray,
    pts_user: np.ndarray,
    joint_errors: Dict[int, float],
    kicking_side: str,
) -> None:
    neutral = (235, 120, 20)  # default blue-ish when no angle available
    joint_colors: Dict[int, Tuple[int, int, int]] = {}
    for j in MAIN_JOINT_TRIPLETS:
        e = joint_errors.get(j, 0.0)
        joint_colors[j] = _error_deg_to_bgr(e)

    # Optional emphasis: kicking-side leg gets slight brightness bump for readability.
    kick_idxs = (24, 26, 28) if kicking_side == "RIGHT" else (23, 25, 27)
    for j in kick_idxs:
        if j in joint_colors:
            b, g, r = joint_colors[j]
            joint_colors[j] = (min(255, b + 10), min(255, g + 10), min(255, r + 10))

    for a, b in HEATMAP_BONES:
        if not _finite_pt(pts_user, a) or not _finite_pt(pts_user, b):
            continue
        if abs(pts_user[a, 0]) + abs(pts_user[a, 1]) < 3 or abs(pts_user[b, 0]) + abs(pts_user[b, 1]) < 3:
            continue
        pa = (int(round(pts_user[a, 0])), int(round(pts_user[a, 1])))
        pb = (int(round(pts_user[b, 0])), int(round(pts_user[b, 1])))
        ca = joint_colors.get(a, neutral)
        cb = joint_colors.get(b, neutral)
        _draw_gradient_bone(img, pa, pb, ca, cb, thickness=4)

    for j, col in joint_colors.items():
        if not _finite_pt(pts_user, j):
            continue
        x, y = int(round(pts_user[j, 0])), int(round(pts_user[j, 1]))
        cv2.circle(img, (x, y), 5, col, -1, cv2.LINE_AA)


def _apply_digital_zoom(
    frame: np.ndarray,
    center_xy: Tuple[float, float],
    zoom: float,
) -> np.ndarray:
    """Digital zoom around center while preserving original frame size."""
    h, w = frame.shape[:2]
    z = float(max(1.0, zoom))
    if z <= 1.001:
        return frame
    cx = float(np.clip(center_xy[0], 0, w - 1))
    cy = float(np.clip(center_xy[1], 0, h - 1))
    crop_w = max(2, int(round(w / z)))
    crop_h = max(2, int(round(h / z)))
    x0 = int(round(cx - crop_w / 2))
    y0 = int(round(cy - crop_h / 2))
    x0 = int(np.clip(x0, 0, max(0, w - crop_w)))
    y0 = int(np.clip(y0, 0, max(0, h - crop_h)))
    crop = frame[y0 : y0 + crop_h, x0 : x0 + crop_w]
    if crop.size == 0:
        return frame
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


def _var_zoom_factor(phase_t: float) -> float:
    """Linear zoom ramp: 1.0 -> 1.5 -> 1.0 through slow-mo window."""
    t = float(np.clip(phase_t, 0.0, 1.0))
    if t < 0.3 or t > 0.7:
        return 1.0
    if t <= 0.4:
        return 1.0 + 0.5 * ((t - 0.3) / 0.1)
    if t <= 0.6:
        return 1.5
    return 1.5 - 0.5 * ((t - 0.6) / 0.1)


def _draw_var_overlay(img: np.ndarray, blink_on: bool) -> None:
    h, w = img.shape[:2]
    if blink_on:
        cv2.putText(
            img,
            "\u2022 VAR ANALYSING",
            (w - 275, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (40, 40, 255),
            2,
            cv2.LINE_AA,
        )
    rec_x = w - 96
    rec_y = 56
    cv2.rectangle(img, (rec_x, rec_y - 18), (rec_x + 72, rec_y + 8), (20, 20, 20), -1)
    cv2.rectangle(img, (rec_x, rec_y - 18), (rec_x + 72, rec_y + 8), (60, 60, 255), 1)
    cv2.circle(img, (rec_x + 12, rec_y - 5), 5, (30, 30, 255), -1, cv2.LINE_AA)
    cv2.putText(img, "REC", (rec_x + 22, rec_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 255), 1, cv2.LINE_AA)


def _torso_lean_deg(pts: np.ndarray) -> Optional[float]:
    req = (11, 12, 23, 24)
    if not all(_finite_pt(pts, i) for i in req):
        return None
    mid_hip = (pts[23] + pts[24]) * 0.5
    mid_sh = (pts[11] + pts[12]) * 0.5
    torso = mid_sh - mid_hip
    if float(np.linalg.norm(torso)) < 1e-6:
        return None
    return abs(float(np.degrees(np.arctan2(torso[0], -torso[1]))))


def _draw_freeze_dashboard(
    img: np.ndarray,
    knee_ext_deg: Optional[float],
    torso_lean_deg: Optional[float],
    form_pct: float,
) -> None:
    h, w = img.shape[:2]
    x0, y0 = int(w * 0.08), int(h * 0.16)
    x1, y1 = int(w * 0.62), int(h * 0.58)
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.62, img, 0.38, 0.0, img)
    cv2.rectangle(img, (x0, y0), (x1, y1), (70, 170, 255), 2)
    cv2.putText(img, "AI COACH ANALYSIS", (x0 + 18, y0 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (220, 240, 255), 2, cv2.LINE_AA)

    knee_txt = "--" if knee_ext_deg is None else f"{knee_ext_deg:.1f}\u00b0"
    lean_txt = "--" if torso_lean_deg is None else f"{torso_lean_deg:.1f}\u00b0"
    lines = [
        f"Knee Extension: {knee_txt}",
        f"Torso Lean: {lean_txt}",
        f"Overall Form Match: {form_pct:.1f}%",
    ]
    status = "PRO FORM" if form_pct >= 82.0 and (knee_ext_deg or 0.0) >= 160.0 and 5.0 <= (torso_lean_deg or 0.0) <= 10.0 else "NEEDS BENDING"
    lines.append(f"Status: {status}")
    y = y0 + 78
    for ln in lines:
        cv2.putText(img, ln, (x0 + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (235, 235, 235), 2, cv2.LINE_AA)
        y += 38

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


def _athlete_bbox_from_pts(pts_user: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    valid: List[Tuple[float, float]] = []
    for i in range(pts_user.shape[0]):
        if not _finite_pt(pts_user, i):
            continue
        valid.append((float(pts_user[i, 0]), float(pts_user[i, 1])))
    if not valid:
        return None
    xs = np.array([p[0] for p in valid], dtype=np.float64)
    ys = np.array([p[1] for p in valid], dtype=np.float64)
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def _storyboard_crop_params(
    frame_w: int,
    frame_h: int,
    pts_user: np.ndarray,
    panel_w: int,
    panel_h: int,
    phase_name: str,
) -> Tuple[int, int, int, int]:
    bbox = _athlete_bbox_from_pts(pts_user)
    if bbox is None:
        return 0, 0, frame_w, frame_h
    x0, y0, x1, y1 = bbox
    bw = max(1.0, x1 - x0)
    bh = max(1.0, y1 - y0)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    target_athlete_ratio = 0.72
    crop_h = int(np.clip(bh / target_athlete_ratio, frame_h * 0.34, frame_h))
    crop_w = int(round(crop_h * (panel_w / max(panel_h, 1))))
    crop_w = int(np.clip(crop_w, frame_w * 0.20, frame_w))
    if phase_name == "impact":
        crop_h = max(2, int(round(crop_h * 0.78)))
        crop_w = max(2, int(round(crop_w * 0.78)))
        if _finite_pt(pts_user, 27) and _finite_pt(pts_user, 28):
            cx = float((pts_user[27, 0] + pts_user[28, 0]) * 0.5)
            cy = float((pts_user[27, 1] + pts_user[28, 1]) * 0.5)
    x = int(round(cx - crop_w * 0.5))
    y = int(round(cy - crop_h * 0.5))
    x = int(np.clip(x, 0, max(0, frame_w - crop_w)))
    y = int(np.clip(y, 0, max(0, frame_h - crop_h)))
    return x, y, crop_w, crop_h


def _panel_verdict(knee_err: float, lean_err: float, impact_speed_px: float) -> str:
    verdicts: List[str] = []
    if knee_err > 15.0:
        verdicts.append("Increase Knee Extension")
    if lean_err < 5.0:
        verdicts.append("Optimal Torso Lean")
    if impact_speed_px >= 700.0:
        verdicts.append("High Velocity Strike")
    if not verdicts:
        verdicts.append("Maintain Posture Consistency")
    return " | ".join(verdicts[:2])


def _build_storyboard(
    report_path: Path,
    snapshots: Dict[str, Dict[str, Any]],
    frame_w: int,
    frame_h: int,
    kicking_side: str,
) -> None:
    panel_h = max(360, frame_h)
    panel_w = max(200, int(round(panel_h * 9.0 / 16.0)))
    bar_h = max(68, int(panel_h * 0.12))
    panel_cards: List[np.ndarray] = []

    for phase_name, _target_t in STORYBOARD_PHASE_TARGETS:
        snap = snapshots.get(phase_name)
        if not snap:
            placeholder = np.zeros((panel_h + bar_h, panel_w, 3), dtype=np.uint8)
            cv2.putText(placeholder, phase_name.upper(), (18, panel_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2, cv2.LINE_AA)
            panel_cards.append(placeholder)
            continue

        frame_raw = snap["frame_raw"].copy()
        pts_user = snap["pts_user"].copy()
        pts_ideal = snap["pts_ideal"].copy()
        x, y, cw, ch = _storyboard_crop_params(frame_w, frame_h, pts_user, panel_w, panel_h, phase_name)
        crop = frame_raw[y : y + ch, x : x + cw]
        panel = cv2.resize(crop, (panel_w, panel_h), interpolation=cv2.INTER_CUBIC)

        sx = panel_w / max(cw, 1)
        sy = panel_h / max(ch, 1)
        pts_user_panel = np.zeros_like(pts_user)
        pts_ideal_panel = np.zeros_like(pts_ideal)
        pts_user_panel[:, 0] = (pts_user[:, 0] - x) * sx
        pts_user_panel[:, 1] = (pts_user[:, 1] - y) * sy
        pts_ideal_panel[:, 0] = (pts_ideal[:, 0] - x) * sx
        pts_ideal_panel[:, 1] = (pts_ideal[:, 1] - y) * sy

        # Draw heatmap after crop/zoom for crisp lines.
        panel_joint_errors = _joint_error_map_deg(pts_user_panel, pts_ideal_panel)
        if phase_name == "impact":
            panel_joint_errors = {k: float(min(JOINT_ERROR_MAX_DEG, v * 1.45)) for k, v in panel_joint_errors.items()}
        _draw_heatmap_skeleton(panel, pts_user_panel, panel_joint_errors, kicking_side)

        knee_idx = 26 if kicking_side == "RIGHT" else 25
        hip_idx = 24 if kicking_side == "RIGHT" else 23
        ank_idx = 28 if kicking_side == "RIGHT" else 27
        ideal_knee = _joint_angle_deg(pts_ideal_panel, hip_idx, knee_idx, ank_idx)
        user_knee = _joint_angle_deg(pts_user_panel, hip_idx, knee_idx, ank_idx)
        knee_err = 0.0 if ideal_knee is None or user_knee is None else _angle_abs_diff_deg(user_knee, ideal_knee)
        user_lean = _torso_lean_deg(pts_user_panel) or 0.0
        lean_err = abs(user_lean - _LEAN_TARGET_DEG)
        verdict = _panel_verdict(knee_err, lean_err, float(snap.get("impact_speed_px", 0.0)))

        bar = np.zeros((bar_h, panel_w, 3), dtype=np.uint8)
        cv2.rectangle(bar, (0, 0), (panel_w - 1, bar_h - 1), (8, 8, 8), -1)
        cv2.rectangle(bar, (0, 0), (panel_w - 1, bar_h - 1), (90, 90, 90), 1)
        cv2.putText(bar, verdict, (10, int(bar_h * 0.62)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 1, cv2.LINE_AA)

        card = np.vstack([panel, bar])
        panel_cards.append(card)

    strip = cv2.hconcat(panel_cards)
    header_h = 54
    footer_h = 48
    report = np.zeros((header_h + strip.shape[0] + footer_h, strip.shape[1], 3), dtype=np.uint8)
    report[:header_h] = (12, 12, 12)
    report[header_h : header_h + strip.shape[0]] = strip
    report[header_h + strip.shape[0] :] = (10, 10, 10)
    cv2.putText(
        report,
        "AMD INSTINCT™ BIOMECHANICAL REPORT | STRIKE LAB",
        (14, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (236, 236, 236),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        report,
        "ANALYZED VIA ROCm ON AMD MI300X INSTINCT GPU",
        (14, report.shape[0] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(report_path), report):
        raise RuntimeError(f"Failed to write storyboard report: {report_path}")


def process_video(
    input_path: Path,
    output_path: Path,
    min_detection: float = 0.5,
    min_tracking: float = 0.5,
) -> Dict[str, str]:
    mp_pose = mp.solutions.pose
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
    seq, vis = _stabilize_subject_track(seq, vis)

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
    kicking_side = _detect_kicking_leg(smooth, vis, peak_idx)

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

    report_path = output_path.with_name(output_path.stem + "_storyboard.png")
    snapshots: Dict[str, Dict[str, Any]] = {
        k: {"dist": float("inf")} for k, _ in STORYBOARD_PHASE_TARGETS
    }

    fi = 0
    out_count = 0
    base_skel_color = (190, 90, 30)  # kept for optional fallback only
    freeze_source: Optional[np.ndarray] = None
    freeze_form_pct = 0.0
    freeze_knee_ext: Optional[float] = None
    freeze_torso_lean: Optional[float] = None
    kick_knee_idx = 26 if kicking_side == "RIGHT" else 25
    kick_hip_idx = 24 if kicking_side == "RIGHT" else 23
    kick_ank_idx = 28 if kicking_side == "RIGHT" else 27
    phase_form_scores: List[float] = []
    phase_torso_leans: List[float] = []
    phase_knee_angles: List[Tuple[float, float]] = []

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

        # Draw only one skeleton layer (heatmap) to avoid duplicate visual outlines.

        in_phase = _in_strike_phase(fi, phase_start, phase_end)
        phase_t = _phase_t(fi, phase_start, phase_end) if in_phase else 0.0
        form_pct = 0.0
        if in_phase:
            ideal_norm = _generate_corrected_ghost(u_norm, v_row, phase_t, kicking_side)
            pts_ideal = np.zeros((33, 2), dtype=np.float64)
            for lm in range(33):
                pts_ideal[lm, 0] = ideal_norm[lm, 0] * w
                pts_ideal[lm, 1] = ideal_norm[lm, 1] * h

            joint_errors = _joint_error_map_deg(pts_user, pts_ideal)
            _draw_heatmap_skeleton(out, pts_user, joint_errors, kicking_side)
            form_pct = _form_match_from_joint_errors(joint_errors)
            phase_form_scores.append(form_pct)
            tl = _torso_lean_deg(pts_user)
            if tl is not None:
                phase_torso_leans.append(float(tl))
            ka = _joint_angle_deg(pts_user, kick_hip_idx, kick_knee_idx, kick_ank_idx)
            if ka is not None:
                phase_knee_angles.append((phase_t, float(ka)))

            # Capture best frames for storyboard targets.
            for phase_name, target_t in STORYBOARD_PHASE_TARGETS:
                d = abs(float(phase_t) - target_t)
                if d < float(snapshots[phase_name]["dist"]):
                    snapshots[phase_name] = {
                        "dist": d,
                        "frame_raw": frame.copy(),
                        "pts_user": pts_user.copy(),
                        "pts_ideal": pts_ideal.copy(),
                        "impact_speed_px": float(speeds_norm[fi] * float(np.hypot(w, h))),
                        "form_pct": float(form_pct),
                    }
            # Keep a dashboard snapshot around impact.
            if 0.45 <= phase_t <= 0.55:
                freeze_source = out.copy()
                freeze_form_pct = form_pct
                freeze_knee_ext = _joint_angle_deg(pts_user, kick_hip_idx, kick_knee_idx, kick_ank_idx)
                freeze_torso_lean = _torso_lean_deg(pts_user)
        else:
            # Outside strike phase, keep a single-color neutral heatmap skeleton.
            _draw_heatmap_skeleton(out, pts_user, {}, kicking_side)

        # Power meter from normalized speed → px/s equivalent for display
        speed_px_equiv = speeds_norm[fi] * float(np.hypot(w, h))
        fill = min(1.0, speed_px_equiv / POWER_METER_REF_SPEED)
        _draw_power_meter(out, fill)

        # Small label near the kicking leg with angle-error-based form score.
        lbl = f"Form Match {form_pct:.0f}%"
        if not in_phase:
            lbl = "Form Match --"
        knee_i = 26 if kicking_side == "RIGHT" else 25
        ankle_i = 28 if kicking_side == "RIGHT" else 27
        if _finite_pt(pts_user, knee_i):
            tx, ty = int(round(pts_user[knee_i, 0])) + 10, int(round(pts_user[knee_i, 1])) - 10
        elif _finite_pt(pts_user, ankle_i):
            tx, ty = int(round(pts_user[ankle_i, 0])) + 10, int(round(pts_user[ankle_i, 1])) - 10
        else:
            tx, ty = 16, h - 24
        tw, th = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
        tx = int(np.clip(tx, 8, max(8, w - tw - 12)))
        ty = int(np.clip(ty, th + 8, max(th + 8, h - 8)))
        cv2.rectangle(out, (tx - 6, ty - th - 6), (tx + tw + 6, ty + 6), (18, 18, 18), -1)
        cv2.rectangle(out, (tx - 6, ty - th - 6), (tx + tw + 6, ty + 6), (255, 180, 40), 1)
        cv2.putText(
            out,
            lbl,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 250, 255),
            2,
            cv2.LINE_AA,
        )
        # Cinematic replay phases:
        # phase_t in [0.3, 0.7] -> 0.2x visual speed via frame repetition + zoom.
        is_slowmo = in_phase and 0.3 <= phase_t <= 0.7
        repeat_n = 5 if is_slowmo else 1
        for _ in range(repeat_n):
            frame_to_write = out.copy()
            if is_slowmo:
                zoom = _var_zoom_factor(phase_t)
                if _finite_pt(pts_user, kick_knee_idx) and _finite_pt(pts_user, kick_ank_idx):
                    zx = float((pts_user[kick_knee_idx, 0] + pts_user[kick_ank_idx, 0]) * 0.5)
                    zy = float((pts_user[kick_knee_idx, 1] + pts_user[kick_ank_idx, 1]) * 0.5)
                elif _finite_pt(pts_user, kick_ank_idx):
                    zx, zy = float(pts_user[kick_ank_idx, 0]), float(pts_user[kick_ank_idx, 1])
                else:
                    zx, zy = w * 0.5, h * 0.55
                frame_to_write = _apply_digital_zoom(frame_to_write, (zx, zy), zoom)
                blink_on = ((out_count // 4) % 2 == 0)
                _draw_var_overlay(frame_to_write, blink_on)
            writer.write(frame_to_write)
            out_count += 1
            freeze_source = frame_to_write.copy()
            freeze_form_pct = form_pct
            if freeze_knee_ext is None:
                freeze_knee_ext = _joint_angle_deg(pts_user, kick_hip_idx, kick_knee_idx, kick_ank_idx)
            if freeze_torso_lean is None:
                freeze_torso_lean = _torso_lean_deg(pts_user)
        fi += 1

    cap.release()
    # Final freeze frame: 2 seconds at nominal FPS.
    if freeze_source is None:
        freeze_source = np.zeros((h, w, 3), dtype=np.uint8)
    freeze = freeze_source.copy()
    _draw_freeze_dashboard(freeze, freeze_knee_ext, freeze_torso_lean, freeze_form_pct)
    freeze_frames = max(1, int(round(fps * 2.0)))
    for _ in range(freeze_frames):
        writer.write(freeze)
        out_count += 1
    writer.release()
    logger.info("Wrote %s (%d source frames -> %d output frames)", output_path, fi, out_count)
    _try_ffmpeg_browser_mp4(output_path)
    _build_storyboard(report_path, snapshots, w, h, kicking_side)
    logger.info("Wrote storyboard report %s", report_path)
    overall_form = float(np.mean(phase_form_scores)) if phase_form_scores else float(freeze_form_pct)
    impact_speed = float(speeds_norm[peak_idx] * float(np.hypot(w, h))) if 0 <= peak_idx < speeds_norm.shape[0] else 0.0
    backswing_vals = [ang for t, ang in phase_knee_angles if t <= 0.35]
    max_backswing_angle = float(max(backswing_vals)) if backswing_vals else 0.0
    impact_window = [ang for t, ang in phase_knee_angles if abs(t - 0.5) <= 0.08]
    knee_extension_at_impact = float(np.mean(impact_window)) if impact_window else float(freeze_knee_ext or 0.0)
    torso_stability = float(np.std(phase_torso_leans)) if phase_torso_leans else 0.0
    mean_lean = float(np.mean(phase_torso_leans)) if phase_torso_leans else float(freeze_torso_lean or 0.0)
    lean_error = abs(mean_lean - _LEAN_TARGET_DEG)
    key_stats = {
        "impact_speed": impact_speed,
        "max_backswing_angle": max_backswing_angle,
        "torso_stability": torso_stability,
        "knee_extension_at_impact": knee_extension_at_impact,
        "overall_form_score": overall_form,
        "lean_error": lean_error,
    }
    coaching_data = generate_coach_verdict(key_stats)
    coaching_data["overall_form_score"] = round(overall_form, 1)
    coaching_data["key_stats"] = {
        "impact_speed": round(impact_speed, 1),
        "max_backswing_angle": round(max_backswing_angle, 1),
        "torso_stability": round(torso_stability, 2),
    }
    return {"video_path": str(output_path), "report_path": str(report_path), "coaching_data": coaching_data}


def main() -> None:
    p = argparse.ArgumentParser(description="Football strike video processor with biomechanical heatmap overlay.")
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
