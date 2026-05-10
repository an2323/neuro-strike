#!/usr/bin/env bash
#
# NeuroStrike — upload project files to the AMD server via scp.
#
# Usage (from repo root):
#   ./upload.sh
#   ./upload.sh 165.245.131.174          # explicit host (overrides default IP)
#   NEUROSTRIKE_SERVER_IP=165.245.133.50 ./upload.sh
#   SSH_KEY=~/.ssh/id_ed25519_amd ./upload.sh
#
# Remote path is fixed: root@<IP>:/root/
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

SERVER_USER="${NEUROSTRIKE_SERVER_USER:-root}"
SERVER_PATH="${NEUROSTRIKE_SERVER_PATH:-/root}"
# First argument overrides IP; else env; else default below.
SERVER_IP="${1:-${NEUROSTRIKE_SERVER_IP:-165.245.133.50}}"

# Optional identity file (common for this project).
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519_amd}"
EXPANDED_KEY="${SSH_KEY/#\~/$HOME}"
SCP_EXTRA=()
if [[ -f "$EXPANDED_KEY" ]]; then
  SCP_EXTRA+=(-i "$EXPANDED_KEY")
fi

DEST="${SERVER_USER}@${SERVER_IP}:${SERVER_PATH}/"

# Root-level Python modules + deps + client + docs
FILES=(
  remote_main.py
  main.py
  app.py
  pose_gpu.py
  strike_video_processor.py
  requirements_remote.txt
  requirements.txt
  requirements_strike_video.txt
  index.html
  DEPLOYMENT.md
)

echo "NeuroStrike — SCP upload"
echo "=========================="
echo "Target: ${SERVER_USER}@${SERVER_IP}:${SERVER_PATH}/"
if ((${#SCP_EXTRA[@]})); then
  echo "SSH key: $EXPANDED_KEY"
fi
echo ""

for f in "${FILES[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "Error: missing required file: $f (run from repo root)" >&2
    exit 1
  fi
done

echo "Uploading ${#FILES[@]} file(s) to ${SERVER_PATH}/..."
scp "${SCP_EXTRA[@]}" "${FILES[@]}" "$DEST"

# Flask Strike Lab expects ./templates and ./static next to app.py
if [[ -d templates ]]; then
  echo "Uploading templates/ ..."
  scp "${SCP_EXTRA[@]}" -r templates "$DEST"
fi
if [[ -d static ]]; then
  echo "Uploading static/ ..."
  scp "${SCP_EXTRA[@]}" -r static "$DEST"
fi
if [[ -d scripts ]]; then
  echo "Uploading scripts/ ..."
  scp "${SCP_EXTRA[@]}" -r scripts "$DEST"
fi

echo ""
echo "Done. Core files on server:"
for f in "${FILES[@]}"; do
  echo "  - ${SERVER_PATH}/${f}"
done
echo "  - ${SERVER_PATH}/templates/ (if present)"
echo "  - ${SERVER_PATH}/static/ (if present)"
echo "  - ${SERVER_PATH}/scripts/ (if present; run scripts/fix_venv_opencv_numpy.sh after pip)"
echo ""
if ((${#SCP_EXTRA[@]})); then
  echo "SSH:  ssh -i $EXPANDED_KEY ${SERVER_USER}@${SERVER_IP}"
else
  echo "SSH:  ssh ${SERVER_USER}@${SERVER_IP}"
fi
echo "Remote WS backend:  cd ${SERVER_PATH} && source neurostrike_env/bin/activate && python3 remote_main.py"
echo "Strike Lab (Flask):  pip install -r requirements.txt && bash scripts/fix_venv_opencv_numpy.sh && python3 app.py"
echo ""
