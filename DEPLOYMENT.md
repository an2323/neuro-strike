# NeuroStrike AMD MI300X Remote Deployment

## Pre-flight Instructions — Run these commands on the AMD server

### 1. SSH into the AMD Server

```bash
ssh root@165.245.128.59
```

### 2. Update System & Install Dependencies

```bash
# Update package lists
apt-get update

# Install Python 3, pip, venv, and build tools
apt-get install -y python3 python3-pip python3-venv build-essential cmake

# Install ROCm drivers (if not already installed for MI300X)
# Verify ROCm installation:
rocm-smi
# If ROCm is not installed, follow: https://rocm.docs.amd.com/en/latest/deploy/linux/install.html
```

### 3. Verify Files Were Uploaded

```bash
# Check that upload.sh ran successfully
ls -la /root/remote_main.py /root/requirements_remote.txt
```

### 4. Create Python Virtual Environment & Install Packages

```bash
# Navigate to the upload directory
cd /root

# Create a virtual environment
python3 -m venv neurostrike_env

# Activate it
source neurostrike_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install all dependencies (Linux-optimized for AMD MI300X)
pip install -r requirements_remote.txt
```

### 5. Verify MediaPipe Works with ROCm GPU

```bash
# Quick test that MediaPipe can initialize
python3 -c "
import mediapipe as mp
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=2)
print('✓ MediaPipe initialized successfully with model_complexity=2')
pose.close()
"
```

### 6. Open Firewall Port (if ufw is active)

```bash
# Allow incoming connections on port 8080
ufw allow 8080/tcp
ufw status
```

### 7. Start the Backend Server

**Option A — Run in tmux session (recommended for persistence):**

```bash
# Install tmux if not available
apt-get install -y tmux

# Create a new tmux session
tmux new-session -s neurostrike

# Inside tmux, activate venv and start the server
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

# Enable and start the service
systemctl daemon-reload
systemctl enable neurostrike.service
systemctl start neurostrike.service

# Check status
systemctl status neurostrike.service
```

### 8. Verify the Server is Running

```bash
# Check the health endpoint
curl http://localhost:8080/health

# Expected response:
# {"status":"healthy","analyzer_ready":true,"active_sessions":0,"max_sessions":10,"timestamp":...}

# Check the root endpoint
curl http://localhost:8080/

# Expected response includes:
# {"service":"NeuroStrike Remote Backend","status":"online","model_complexity":2,...}
```

### 9. Monitor Logs

```bash
# If using systemd:
journalctl -u neurostrike.service -f

# If using nohup:
tail -f /root/neurostrike.log

# If using tmux: simply attach to the session
tmux attach -t neurostrike
```

### 10. Connect from the Frontend

1. Open `index.html` in a browser (served locally or from any web server)
2. The **AMD Server IP** field is pre-filled with `165.245.128.59`
3. Click **Connect** — the indicator should turn green and show "Connected to 165.245.128.59"
4. Click **Start Training** to begin sending frames to the AMD server

---

## Architecture Overview

```
┌─────────────────────┐         WebSocket          ┌──────────────────────────┐
│   Browser (index.html)  │ ──────────────────────▶ │  AMD MI300X Server       │
│                         │ ◀────────────────────── │  remote_main.py          │
│  • Camera/Video Input   │    ws://IP:8080/ws      │  • MediaPipe Pose (GPU)  │
│  • Skeleton Overlay     │                         │  • model_complexity=2    │
│  • Latency Monitor      │                         │  • Connection Pool (max) │
│  • AMD Server Connect   │                         │  • Per-session Queue     │
└─────────────────────┘                           └──────────────────────────┘
```

## Configuration Reference

| Setting          | Value             | Description                       |
| ---------------- | ----------------- | --------------------------------- |
| Server IP        | `165.245.128.59` | AMD MI300X server address         |
| Port             | `8080`            | WebSocket & HTTP port             |
| Model Complexity | `2`               | Maximum MediaPipe precision       |
| GPU              | ROCm (MI300X)     | AMD GPU acceleration              |
| Max Sessions     | `10`              | Concurrent connection limit       |
| Session Queue    | `8` frames        | Per-connection frame buffer       |
| Frame Rate Limit | `33 FPS`          | Max frames per second per session |
