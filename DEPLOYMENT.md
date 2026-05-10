# NeuroStrike AMD MI300X Remote Deployment

## Pre-flight Instructions — Run these commands on the AMD server

### 1. SSH into the AMD Server

```bash
ssh root@165.245.133.50
```

---

### 2. Update System & Install Core Dependencies

```bash
apt-get update
apt-get install -y python3 python3-pip python3-venv build-essential cmake wget gnupg2 ffmpeg
```

---

### 3. Install ROCm 6.x + MIGraphX (AMD GPU runtime)

MI300X requires ROCm 6.x. MIGraphX is AMD's native inference engine used by
`onnxruntime-rocm` to run pose estimation on the GPU.

```bash
# Add AMD GPG key
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
  gpg --dearmor -o /etc/apt/keyrings/rocm.gpg

# Add ROCm 6.2 repo (Ubuntu 22.04 / jammy — adjust codename if on a different distro)
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] \
  https://repo.radeon.com/rocm/apt/6.2 jammy main" \
  | tee /etc/apt/sources.list.d/rocm.list

# Install ROCm runtime + MIGraphX
apt-get update
apt-get install -y rocm-hip-runtime rocm-dev rocm-smi-lib migraphx half

# Add root to GPU device groups
usermod -aG render,video root

# Verify hardware is visible
rocm-smi
# Expected: table showing AMD Instinct MI300X

rocminfo | grep -i "name"
# Expected: lines including "gfx942"
```

---

### 4. Verify Files Were Uploaded

```bash
# After running ./upload.sh on your local machine:
ls -la /root/remote_main.py /root/requirements_remote.txt /root/pose_gpu.py \
        /root/scripts/download_blazepose_onnx.sh
```

---

### 5. Create Python Virtual Environment & Install Packages

```bash
cd /root
python3 -m venv neurostrike_env
source neurostrike_env/bin/activate
pip install --upgrade pip

# Install base deps (mediapipe stays as CPU fallback)
pip install -r requirements_remote.txt

# Install onnxruntime with ROCm + MIGraphX EP (replaces plain onnxruntime)
# Use the repo matching the server's ROCm version (check: cat /opt/rocm/.info/version)
pip install onnxruntime-rocm \
  --extra-index-url https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0/

# Fix OpenCV / NumPy conflicts if needed
bash scripts/fix_venv_opencv_numpy.sh
```

---

### 6. ROCm 7.0 hipBLAS Compatibility Shim (already applied on current droplet)

`onnxruntime-rocm` references two hipBLAS symbols (`hipblasGemmEx_v2`,
`hipblasGemmStridedBatchedEx_v2`) that were renamed in ROCm 7.0. A thin
forwarding shim is required to bridge the gap.

```bash
mkdir -p /root/lib/rocm-compat

# SO-version symlinks (onnxruntime expects .so.6/.so.2, ROCm 7.0 ships .so.7/.so.3)
ln -sf /opt/rocm/lib/libamdhip64.so.7  /root/lib/rocm-compat/libamdhip64.so.6
ln -sf /opt/rocm/lib/libhipblas.so.3   /root/lib/rocm-compat/libhipblas.so.2

# Compile the _v2 symbol shim (forwards to ROCm 7.0 base functions)
cat > /tmp/hipblas_shim.c << 'EOF'
#include <hipblas/hipblas.h>
hipblasStatus_t hipblasGemmEx_v2(
    hipblasHandle_t h, hipblasOperation_t ta, hipblasOperation_t tb,
    int m, int n, int k, const void* alpha,
    const void* A, hipDataType at, int lda,
    const void* B, hipDataType bt, int ldb,
    const void* beta,
    void* C, hipDataType ct, int ldc,
    hipDataType ct2, hipblasGemmAlgo_t algo) {
    return hipblasGemmEx(h,ta,tb,m,n,k,alpha,A,at,lda,B,bt,ldb,beta,C,ct,ldc,ct2,algo);
}
hipblasStatus_t hipblasGemmStridedBatchedEx_v2(
    hipblasHandle_t h, hipblasOperation_t ta, hipblasOperation_t tb,
    int m, int n, int k, const void* alpha,
    const void* A, hipDataType at, int lda, long long int sa,
    const void* B, hipDataType bt, int ldb, long long int sb,
    const void* beta,
    void* C, hipDataType ct, int ldc, long long int sc,
    int bc, hipDataType ct2, hipblasGemmAlgo_t algo) {
    return hipblasGemmStridedBatchedEx(h,ta,tb,m,n,k,alpha,A,at,lda,sa,B,bt,ldb,sb,beta,C,ct,ldc,sc,bc,ct2,algo);
}
EOF
gcc -shared -fPIC -O2 -o /root/lib/rocm-compat/libhipblas_shim.so /tmp/hipblas_shim.c \
  -I/opt/rocm/include -L/opt/rocm/lib -lhipblas -Wl,-rpath,/opt/rocm/lib

# Inject into venv activate (idempotent)
grep -q 'rocm-compat' /root/neurostrike_env/bin/activate || cat >> /root/neurostrike_env/bin/activate << 'ENVEOF'
export LD_LIBRARY_PATH="/root/lib/rocm-compat:/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="/root/lib/rocm-compat/libhipblas_shim.so${LD_PRELOAD:+:$LD_PRELOAD}"
ENVEOF
```

---

### 7. Download BlazePose ONNX Models

```bash
cd /root
source neurostrike_env/bin/activate
bash scripts/download_blazepose_onnx.sh
# Expected output: two model files in models/
#   models/pose_detection.onnx       (~2 MB)
#   models/pose_landmark_heavy.onnx  (~25 MB)
```

---

### 7. Verify GPU Inference Stack

```bash
source neurostrike_env/bin/activate

# Check MIGraphX EP is available in onnxruntime
python3 -c "
import onnxruntime as ort
providers = ort.get_available_providers()
print('Providers:', providers)
assert 'MIGraphXExecutionProvider' in providers, 'MIGraphX EP not found — check ROCm install'
print('OK: MIGraphX EP available')
"

# Load BlazePoseONNX and run a smoke test
python3 -c "
import numpy as np
from pose_gpu import BlazePoseONNX
bp = BlazePoseONNX()
print('MIGraphX active:', bp.using_migraphx)
dummy = np.zeros((480, 640, 3), dtype=np.uint8)
res = bp.process(dummy)
print('Smoke test OK (pose_landmarks:', res.pose_landmarks, ')')
"
```

---

### 8. Open Firewall Port (if ufw is active)

```bash
ufw allow 8080/tcp
ufw status
```

---

### 9. Start the Backend Server

**Option A — Run in tmux session (recommended for persistence):**

```bash
apt-get install -y tmux
tmux new-session -s neurostrike

# Inside tmux:
cd /root
source neurostrike_env/bin/activate
python3 remote_main.py
```

**Option B — Run with nohup (simple background):**

```bash
cd /root
source neurostrike_env/bin/activate
nohup python3 remote_main.py > neurostrike.log 2>&1 &
echo "Server PID: $!"
```

**Option C — Install as systemd service (auto-restart on reboot):**

```bash
cat > /etc/systemd/system/neurostrike.service << 'EOF'
[Unit]
Description=NeuroStrike Remote Backend (AMD MI300X)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root
Environment="MEDIAPIPE_DISABLE_GPU=0"
ExecStart=/root/neurostrike_env/bin/python3 /root/remote_main.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable neurostrike.service
systemctl start neurostrike.service
systemctl status neurostrike.service
```

---

### 10. Verify the Server is Running

```bash
# Standard health check
curl http://localhost:8080/health
# Expected: {"status":"healthy","analyzer_ready":true,...}

# GPU status (new endpoint — confirms ROCm is actually used)
curl http://localhost:8080/gpu-status
# Expected: {"rocm_available":true,"migraphx_provider":true,"pose_backend":"blazepose-onnx-MIGraphX",...}

# Root endpoint
curl http://localhost:8080/
# Expected: {...,"gpu_enabled":true,"gpu_backend":"ROCm MIGraphX (AMD MI300X)",...}
```

---

### 11. Monitor Logs

```bash
# systemd:
journalctl -u neurostrike.service -f

# nohup:
tail -f /root/neurostrike.log

# tmux:
tmux attach -t neurostrike
```

---

### 12. Connect from the Frontend

1. Open `index.html` in a browser (served locally or from any web server)
2. The **AMD Server IP** field is pre-filled with `165.245.133.50`
3. Click **Connect** — the indicator should turn green and show "Connected to 165.245.133.50"
4. Click **Start Training** to begin sending frames to the AMD server

---

## Architecture Overview

```
┌─────────────────────┐         WebSocket          ┌──────────────────────────────────┐
│   Browser (index.html)  │ ──────────────────────▶ │  AMD MI300X Server               │
│                         │ ◀────────────────────── │  remote_main.py                  │
│  • Camera/Video Input   │    ws://IP:8080/ws      │  • BlazePose ONNX (MIGraphX GPU) │
│  • Skeleton Overlay     │                         │  • MediaPipe CPU fallback         │
│  • Latency Monitor      │                         │  • Connection Pool (max 10)       │
│  • AMD Server Connect   │                         │  • Per-session Queue              │
└─────────────────────┘                           └──────────────────────────────────┘
```

## Configuration Reference

| Setting          | Value             | Description                            |
| ---------------- | ----------------- | -------------------------------------- |
| Server IP        | `165.245.133.50`  | AMD MI300X server address              |
| Port             | `8080`            | WebSocket & HTTP port                  |
| Pose backend     | BlazePose ONNX    | MIGraphX EP (GPU) → MediaPipe (CPU)    |
| Model Complexity | `2` (heavy)       | Maximum BlazePose precision            |
| GPU              | ROCm / MIGraphX   | AMD GPU acceleration via onnxruntime   |
| Max Sessions     | `10`              | Concurrent connection limit            |
| Session Queue    | `8` frames        | Per-connection frame buffer            |
| Frame Rate Limit | `33 FPS`          | Max frames per second per session      |
