#!/bin/bash
# Idempotent startup script for TPU VMs using HF Buckets

exec > >(tee -a /var/log/tpu_startup.log) 2>&1
echo "Starting TPU setup..."

# 1. Fetch metadata from GCP
get_metadata() {
    curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"
}

export HF_TOKEN=$(get_metadata "HF_TOKEN")
export WANDB_API_KEY=$(get_metadata "WANDB_API_KEY")
export RUN_ID=$(get_metadata "RUN_ID")
export HF_REPO=$(get_metadata "HF_REPO")
export ACCEL_TYPE=$(get_metadata "ACCEL_TYPE")
export GITHUB_REPO=$(get_metadata "GITHUB_REPO")
export BRANCH=$(get_metadata "BRANCH")
export TRAIN_CMD=$(get_metadata "TRAIN_CMD")

export WORKER_ID=$(hostname)

# 2. Install dependencies
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y git tmux jq curl build-essential wget

# Install HF CLI
curl -LsSf https://hf.co/cli/install.sh | bash

# Install Miniconda to ensure modern Python (3.10+) on older TPU OS bases
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p /opt/conda
rm /tmp/miniconda.sh
export PATH="/opt/conda/bin:$PATH"

# Create and activate conda environment
conda create -n phantom python=3.11 -y
source /opt/conda/bin/activate phantom

# Install Python ML dependencies
pip install --upgrade pip
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install wandb orbax-checkpoint huggingface_hub

# 3. Setup directories
mkdir -p /app/data
mkdir -p /app/checkpoints
mkdir -p /app/logs
mkdir -p /app/xla_cache/$ACCEL_TYPE

export JAX_COMPILATION_CACHE_DIR="/app/xla_cache/${ACCEL_TYPE}"

# 4. Clone repository
if [ -d "/app/model" ]; then
    rm -rf /app/model
fi
git clone --branch $BRANCH $GITHUB_REPO /app/model
cd /app/model

# Install project-specific dependencies if available
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi
if [ -f "sim/requirements.txt" ]; then
    pip install -r sim/requirements.txt
fi

# 5. Restore state from Hugging Face Buckets
echo "Restoring state from hf://buckets/$HF_REPO..."
# Download base data (shared across all)
hf buckets sync hf://buckets/$HF_REPO/data/base /app/data || echo "No base data found or failed to sync."

# Download worker-specific checkpoints and logs
hf buckets sync hf://buckets/$HF_REPO/runs/$RUN_ID/checkpoints/$WORKER_ID /app/checkpoints || echo "No checkpoint found."
hf buckets sync hf://buckets/$HF_REPO/runs/$RUN_ID/logs/$WORKER_ID /app/logs || echo "No logs found."

# Download architecture-specific XLA cache
hf buckets sync hf://buckets/$HF_REPO/runs/$RUN_ID/xla/$ACCEL_TYPE /app/xla_cache/$ACCEL_TYPE || echo "No XLA cache found."

# 6. Start Background Sync Loop
cat << 'EOF' > /app/sync_loop.sh
#!/bin/bash
while true; do
  sleep 120
  echo "[$(date)] Background sync to HF Bucket..."
  hf buckets sync /app/checkpoints hf://buckets/$HF_REPO/runs/$RUN_ID/checkpoints/$WORKER_ID --quiet || true
  hf buckets sync /app/logs hf://buckets/$HF_REPO/runs/$RUN_ID/logs/$WORKER_ID --quiet || true
  hf buckets sync /app/xla_cache/$ACCEL_TYPE hf://buckets/$HF_REPO/runs/$RUN_ID/xla/$ACCEL_TYPE --quiet || true
done
EOF
chmod +x /app/sync_loop.sh
/app/sync_loop.sh &
SYNC_PID=$!

# 7. Execute Training
echo "Starting training with command: $TRAIN_CMD"
# Ensure we are in the correct directory and environment
cd /app/model
export PYTHONPATH="/app/model:$PYTHONPATH"

if [ -n "$TRAIN_CMD" ]; then
    eval "$TRAIN_CMD"
    EXIT_CODE=$?
else
    echo "No TRAIN_CMD provided. Sleeping for testing purposes..."
    # For testing: run a dummy process so the VM doesn't just idle immediately
    sleep 3600
    EXIT_CODE=0
fi

# 8. Cleanup and Final Sync
echo "Training finished with exit code $EXIT_CODE. Stopping sync loop and performing final sync..."
kill $SYNC_PID

hf buckets sync /app/checkpoints hf://buckets/$HF_REPO/runs/$RUN_ID/checkpoints/$WORKER_ID
hf buckets sync /app/logs hf://buckets/$HF_REPO/runs/$RUN_ID/logs/$WORKER_ID
hf buckets sync /app/xla_cache/$ACCEL_TYPE hf://buckets/$HF_REPO/runs/$RUN_ID/xla/$ACCEL_TYPE

exit $EXIT_CODE