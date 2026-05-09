#!/bin/bash

# NeuroStrike - AMD MI300X Deployment Script
# Uploads backend files to remote server

SERVER_IP="165.245.131.174"
SERVER_USER="root"
SERVER_PATH="/root"

echo "🚀 NeuroStrike AMD MI300X Deployment"
echo "======================================"
echo "Target: ${SERVER_USER}@${SERVER_IP}:${SERVER_PATH}"
echo ""

# Check if files exist
FILES=("remote_main.py" "requirements_remote.txt" "DEPLOYMENT.md")
for f in "${FILES[@]}"; do
    if [ ! -f "$f" ]; then
        echo "❌ Error: $f not found"
        exit 1
    fi
done

echo "📦 Uploading files..."
echo ""

# Upload each file
for f in "${FILES[@]}"; do
    echo "→ Uploading $f..."
    scp "$f" ${SERVER_USER}@${SERVER_IP}:${SERVER_PATH}/
    if [ $? -eq 0 ]; then
        echo "✓ $f uploaded successfully"
    else
        echo "❌ Failed to upload $f"
        exit 1
    fi
done

echo ""
echo "✅ All files uploaded successfully!"
echo ""
echo "📋 Next Steps:"
echo "1. SSH into the server:"
echo "   ssh ${SERVER_USER}@${SERVER_IP}"
echo ""
echo "2. Run setup on the server:"
echo "   cd /root && python3 -m venv neurostrike_env && source neurostrike_env/bin/activate && pip install -r requirements_remote.txt"
echo ""
echo "3. Start the backend:"
echo "   python3 remote_main.py"
echo ""
echo "📖 Full instructions in DEPLOYMENT.md"
echo ""
