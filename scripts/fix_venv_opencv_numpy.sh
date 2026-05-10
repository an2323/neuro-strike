#!/usr/bin/env bash
#
# Repair a broken NumPy / OpenCV stack in the active venv.
#
# Typical causes:
#   1) Corrupt pip state: site-packages/~umpy* (failed uninstall/rename)
#   2) mediapipe pulls opencv-contrib-python, which shadows opencv-python-headless
#      and breaks cv2 against numpy (ImportError: numpy._core.multiarray / _signature_descriptor)
#
# Usage (on the server, venv ACTIVE):
#   source neurostrike_env/bin/activate
#   bash scripts/fix_venv_opencv_numpy.sh
#
set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "Error: activate your venv first, e.g.  source neurostrike_env/bin/activate" >&2
  exit 1
fi

echo "== Removing corrupt pip leftovers (~umpy*) =="
python3 <<'PY'
import shutil
import site
from pathlib import Path

for base in site.getsitepackages():
    root = Path(base)
    if not root.is_dir():
        continue
    for pattern in ("~umpy*", "~umpy*.dist-info"):
        for p in sorted(root.glob(pattern)):
            print("  rm -rf", p)
            shutil.rmtree(p, ignore_errors=True)
PY

echo "== Uninstalling duplicate / GUI OpenCV wheels (keep headless only) =="
pip uninstall -y opencv-contrib-python opencv-python 2>/dev/null || true

echo "== Force-reinstalling numpy + opencv-python-headless (binary stack) =="
pip install --no-cache-dir --force-reinstall "numpy==1.26.4" "opencv-python-headless==4.9.0.80"

echo "== Smoke test =="
python3 -c "import numpy, cv2; print('numpy', numpy.__version__, 'cv2 OK', cv2.__version__)"

echo "== Optional: mediapipe import =="
python3 - <<'PY' || true
try:
    import mediapipe as mp

    print("mediapipe", mp.__version__)
except Exception as exc:
    print("mediapipe (non-fatal):", exc)
PY

echo "Done."
