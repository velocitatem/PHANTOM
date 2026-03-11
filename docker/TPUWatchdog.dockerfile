FROM google/cloud-sdk:slim

# Install tmux to manage multiple watchdogs and jq for json parsing
RUN apt-get update && \
    apt-get install -y tmux jq && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the orchestration scripts and configs
COPY tpu_orchestration/ /app/tpu_orchestration/

# Make sure scripts are executable
RUN chmod +x /app/tpu_orchestration/watchdog.sh
RUN chmod +x /app/tpu_orchestration/tpu_startup.sh

# Create an entrypoint script that launches a watchdog for each config
COPY <<-'EOF' /app/entrypoint.sh
#!/bin/bash
set -e

# Make sure required variables are set
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable is required."
    exit 1
fi

if [ -z "$WANDB_API_KEY" ]; then
    echo "Warning: WANDB_API_KEY environment variable is not set. Wandb logging may fail on TPUs."
fi

# Authenticate gcloud if credentials are provided
if [ -n "$GOOGLE_APPLICATION_CREDENTIALS" ] && [ -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    CRED_TYPE=$(jq -r '.type' "$GOOGLE_APPLICATION_CREDENTIALS" 2>/dev/null || echo "unknown")
    if [ "$CRED_TYPE" = "service_account" ]; then
        echo "Authenticating gcloud using service account key..."
        gcloud auth activate-service-account --key-file="$GOOGLE_APPLICATION_CREDENTIALS"
        
        # Extract project ID from the key file
        PROJECT_ID=$(jq -r '.project_id' "$GOOGLE_APPLICATION_CREDENTIALS")
        if [ -n "$PROJECT_ID" ] && [ "$PROJECT_ID" != "null" ]; then
            gcloud config set project "$PROJECT_ID"
            echo "Set project to $PROJECT_ID"
        fi
    else
        echo "Note: Using application default credentials or mounted gcloud config..."
    fi
else
    echo "Note: Assuming gcloud config is mounted from host."
fi

# Run the watchdogs in the background using bash instead of tmux
# Tmux needs a TTY to attach properly which we might not have in docker
# Stagger startups by 15s to prevent simultaneous TPU creation quota hits
DELAY=0
for conf in /app/tpu_orchestration/configs/*.conf; do
    echo "Starting watchdog for $(basename "$conf" .conf) (delay: ${DELAY}s)"
    (sleep $DELAY && /app/tpu_orchestration/watchdog.sh "$conf") &
    DELAY=$((DELAY + 15))
done

echo "All watchdogs queued with staggered startup."

# Keep the container running
wait
EOF

RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]