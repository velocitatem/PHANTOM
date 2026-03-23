#!/bin/bash
# Watchdog loop to ensure TPUs are re-queued when preempted

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <config_file>"
    exit 1
fi

CONFIG_FILE=$1
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file $CONFIG_FILE not found."
    exit 1
fi

# Load config
source "$CONFIG_FILE"

# Make sure HF_TOKEN is available
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable must be set before running watchdog."
    echo "export HF_TOKEN=..."
    exit 1
fi

# Make sure WANDB_API_KEY is available
if [ -z "$WANDB_API_KEY" ]; then
    echo "Warning: WANDB_API_KEY environment variable is not set. Wandb logging may fail."
fi

# Make sure GITHUB_REPO is set in config or env
if [ -z "$GITHUB_REPO" ]; then
    GITHUB_REPO="https://github.com/velocitatem/PHANTOM.git"
    if [ -n "$GITHUB_TOKEN" ]; then
        GITHUB_REPO="https://velocitatem:${GITHUB_TOKEN}@github.com/velocitatem/PHANTOM.git"
    fi
fi

# Make sure BRANCH is set in config or env
if [ -z "$BRANCH" ]; then
    BRANCH="main"
fi

# Ensure PROJECT_ID is set
if [ -z "$PROJECT_ID" ]; then
    PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
    if [ -z "$PROJECT_ID" ]; then
        PROJECT_ID="phantom-trc" # Fallback to the known project ID
        echo "Warning: PROJECT_ID not set and gcloud not configured. Defaulting to $PROJECT_ID"
    fi
fi

echo "Starting watchdog for $QR_NAME in $ZONE (Project: $PROJECT_ID)"
echo "Accelerator: $ACCEL_TYPE"
echo "Run ID: $RUN_ID"

# Backoff tracking for IP quota errors
RETRY_DELAY=60
MAX_RETRY_DELAY=300

while true; do
  STATE=$(gcloud compute tpus queued-resources describe $QR_NAME --zone=$ZONE --project=$PROJECT_ID --format="value(state)" 2>/dev/null)
  
  if [ -z "$STATE" ] || [[ "$STATE" == *"SUSPENDED"* ]] || [[ "$STATE" == *"FAILED"* ]]; then
    echo "[$(date)] Cluster '${STATE:-MISSING}' - cleaning IPs and re-queuing..."
    
    # Clean all orphaned RESERVED IPs in parallel to free quota
    gcloud compute addresses list --project=$PROJECT_ID \
      --filter="status=RESERVED AND name~'^tpu-.*'" \
      --format="value(name,region)" 2>/dev/null | \
      while IFS=$'\t' read -r n r; do 
        [ -n "$n" ] && [ -n "$r" ] && gcloud compute addresses delete "$n" --region="$r" --project=$PROJECT_ID --quiet 2>/dev/null &
      done
    wait
    
    # Delete QR and any orphaned VM
    gcloud compute tpus queued-resources delete $QR_NAME --zone=$ZONE --project=$PROJECT_ID --quiet --force 2>/dev/null
    VM_STATE=$(gcloud compute tpus tpu-vm describe $QR_NAME --zone=$ZONE --project=$PROJECT_ID --format="value(state)" 2>/dev/null)
    [ -n "$VM_STATE" ] && gcloud compute tpus tpu-vm delete $QR_NAME --zone=$ZONE --project=$PROJECT_ID --quiet 2>/dev/null
    
    sleep 5
    
    # Create new QR
    SPOT_FLAG=""
    if [ "$IS_SPOT" = "true" ]; then
        SPOT_FLAG="--spot"
    fi
    
    # Prepare metadata
    METADATA="HF_TOKEN=$HF_TOKEN,RUN_ID=$RUN_ID,HF_REPO=$HF_REPO,ACCEL_TYPE=$ACCEL_TYPE,GITHUB_REPO=$GITHUB_REPO,BRANCH=$BRANCH"
    if [ -n "$WANDB_API_KEY" ]; then
        METADATA="$METADATA,WANDB_API_KEY=$WANDB_API_KEY"
    fi
    if [ -n "$TRAIN_CMD" ]; then
        METADATA="$METADATA,TRAIN_CMD=$TRAIN_CMD"
    fi

    # Determine runtime version
    RT_VERSION=${RUNTIME_VERSION:-"v2-alpha-tpuv4"}

    gcloud compute tpus queued-resources create $QR_NAME \
      --project=$PROJECT_ID \
      --node-id=$QR_NAME \
      --zone=$ZONE \
      --accelerator-type=$ACCEL_TYPE \
      --runtime-version=$RT_VERSION \
      $SPOT_FLAG \
      --metadata-from-file startup-script=$(dirname $0)/tpu_startup.sh \
      --metadata "$METADATA" 2>&1 | tee /tmp/tpu_create_${QR_NAME}.log
      
    if [ $? -eq 0 ]; then
        echo "[$(date)] Successfully queued $QR_NAME."
        RETRY_DELAY=60
    elif grep -q "IN_USE_ADDRESSES" /tmp/tpu_create_${QR_NAME}.log 2>/dev/null; then
        echo "[$(date)] IP quota hit - backing off ${RETRY_DELAY}s"
        sleep $RETRY_DELAY
        RETRY_DELAY=$((RETRY_DELAY * 2))
        [ $RETRY_DELAY -gt $MAX_RETRY_DELAY ] && RETRY_DELAY=$MAX_RETRY_DELAY
        continue
    else
        echo "[$(date)] Failed to queue $QR_NAME."
        RETRY_DELAY=60
    fi
  else
    echo "[$(date)] Cluster state is $STATE. Checking again in 60s..."
  fi
  sleep 60
done
