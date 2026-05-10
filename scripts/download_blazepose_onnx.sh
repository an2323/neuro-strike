#!/usr/bin/env bash
# download_blazepose_onnx.sh
#
# Downloads BlazePose ONNX models required by pose_gpu.py into models/.
#
# Strategy (tried in order):
#   1. PINTO0309 HuggingFace — pre-converted ONNX (preferred)
#   2. Mediapipe package extraction + tf2onnx conversion (fallback)
#
# Usage (from repo root, venv active):
#   bash scripts/download_blazepose_onnx.sh
#
# Outputs:
#   models/pose_detection.onnx       (128×128 person detector, 896 SSD anchors)
#   models/pose_landmark_heavy.onnx  (256×256 landmark model, 33 keypoints × 5)
#
set -euo pipefail

MODELS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/models"
mkdir -p "$MODELS_DIR"

DET_PATH="$MODELS_DIR/pose_detection.onnx"
LM_PATH="$MODELS_DIR/pose_landmark_heavy.onnx"

echo "=== NeuroStrike: BlazePose ONNX model download ==="
echo "Target directory: $MODELS_DIR"
echo ""

# -----------------------------------------------------------------------
# Helper: download with wget or curl
# -----------------------------------------------------------------------
_download() {
  local url="$1" dest="$2" label="$3"
  echo "  Downloading $label ..."
  if command -v wget &>/dev/null; then
    wget -q --show-progress -O "$dest" "$url" && return 0
  elif command -v curl &>/dev/null; then
    curl -fsSL -o "$dest" "$url" && return 0
  fi
  echo "  ERROR: neither wget nor curl found." >&2
  return 1
}

# -----------------------------------------------------------------------
# Strategy 1: PINTO0309 HuggingFace
# -----------------------------------------------------------------------
HF_BASE="https://huggingface.co/PINTO0309/053_BlazePose/resolve/main"

_try_pinto() {
  echo "--- Strategy 1: PINTO0309 HuggingFace ---"
  local ok=0
  _download \
    "${HF_BASE}/pose_detection.onnx" \
    "$DET_PATH" \
    "pose_detection.onnx" || ok=1
  _download \
    "${HF_BASE}/pose_landmark_heavy.onnx" \
    "$LM_PATH" \
    "pose_landmark_heavy.onnx" || ok=1
  return $ok
}

# -----------------------------------------------------------------------
# Strategy 2: Extract TFLite from mediapipe package, convert with tf2onnx
# -----------------------------------------------------------------------
_try_mediapipe_extract() {
  echo "--- Strategy 2: mediapipe TFLite extraction + tf2onnx conversion ---"

  if ! python3 -c "import mediapipe" 2>/dev/null; then
    echo "  mediapipe not installed — skipping."
    return 1
  fi

  echo "  Locating TFLite models inside mediapipe package..."
  TFLITE_DIR=$(python3 - <<'PY'
import pathlib, mediapipe
mp_root = pathlib.Path(mediapipe.__file__).parent
# Search for the heavy landmark model
for p in mp_root.rglob("pose_landmark_heavy.tflite"):
    print(p.parent)
    break
PY
)

  if [[ -z "$TFLITE_DIR" ]]; then
    echo "  pose_landmark_heavy.tflite not found in mediapipe package."
    return 1
  fi

  echo "  Found TFLite models at: $TFLITE_DIR"
  DET_TFLITE="$TFLITE_DIR/pose_detection.tflite"
  LM_TFLITE="$TFLITE_DIR/pose_landmark_heavy.tflite"

  # Fall back to broader search for detector if not co-located
  if [[ ! -f "$DET_TFLITE" ]]; then
    DET_TFLITE=$(python3 - <<'PY'
import pathlib, mediapipe
mp_root = pathlib.Path(mediapipe.__file__).parent
for p in mp_root.rglob("pose_detection.tflite"):
    print(p); break
PY
)
  fi

  for f in "$DET_TFLITE" "$LM_TFLITE"; do
    if [[ ! -f "$f" ]]; then
      echo "  Missing: $f"
      return 1
    fi
  done

  echo "  Installing tf2onnx for conversion..."
  pip install -q tf2onnx onnx

  echo "  Converting pose_detection.tflite ..."
  python3 -m tf2onnx.convert \
    --tflite "$DET_TFLITE" \
    --output "$DET_PATH" \
    --opset 16 \
    --inputs-as-nchw input 2>/dev/null || \
  python3 -m tf2onnx.convert \
    --tflite "$DET_TFLITE" \
    --output "$DET_PATH" \
    --opset 16

  echo "  Converting pose_landmark_heavy.tflite ..."
  python3 -m tf2onnx.convert \
    --tflite "$LM_TFLITE" \
    --output "$LM_PATH" \
    --opset 16

  echo "  Conversion complete."
  return 0
}

# -----------------------------------------------------------------------
# Main: try strategies in order
# -----------------------------------------------------------------------
need_det=false
need_lm=false

[[ -f "$DET_PATH" ]] && echo "✓ pose_detection.onnx already exists ($(du -h "$DET_PATH" | cut -f1))" || need_det=true
[[ -f "$LM_PATH"  ]] && echo "✓ pose_landmark_heavy.onnx already exists ($(du -h "$LM_PATH" | cut -f1))" || need_lm=true

if ! $need_det && ! $need_lm; then
  echo ""
  echo "Both models already present — nothing to download."
  echo "Delete them manually to force re-download."
  exit 0
fi

echo ""
success=false

if _try_pinto; then
  success=true
else
  echo ""
  echo "Strategy 1 failed — trying Strategy 2 ..."
  echo ""
  if _try_mediapipe_extract; then
    success=true
  fi
fi

echo ""
if $success && [[ -f "$DET_PATH" && -f "$LM_PATH" ]]; then
  echo "=== Download complete ==="
  echo "  pose_detection.onnx      $(du -h "$DET_PATH" | cut -f1)"
  echo "  pose_landmark_heavy.onnx $(du -h "$LM_PATH"  | cut -f1)"
  echo ""
  echo "Smoke-test (from repo root, venv active):"
  echo "  python3 -c \"from pose_gpu import BlazePoseONNX; bp = BlazePoseONNX(); print('OK')\""
else
  echo "=== FAILED: could not obtain ONNX models ==="
  echo ""
  echo "Manual options:"
  echo "  A) Download from PINTO HuggingFace manually:"
  echo "       https://huggingface.co/PINTO0309/053_BlazePose"
  echo "     Place pose_detection.onnx + pose_landmark_heavy.onnx in models/"
  echo ""
  echo "  B) Convert from MediaPipe TFLite yourself:"
  echo "       pip install tf2onnx"
  echo "       Find .tflite files: python3 -c \"import pathlib,mediapipe; [print(p) for p in pathlib.Path(mediapipe.__file__).parent.rglob('*.tflite')]\""
  echo "       python3 -m tf2onnx.convert --tflite <path>.tflite --output models/<name>.onnx --opset 16"
  exit 1
fi
