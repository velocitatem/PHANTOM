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

        if [ -z "$PROJECT_ID" ]; then
            PROJECT_ID=$(jq -r '.project_id // empty' "$GOOGLE_APPLICATION_CREDENTIALS")
        fi
    elif [ "$CRED_TYPE" = "authorized_user" ]; then
        echo "Using authorized_user credentials via credential file override..."
        export CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE="$GOOGLE_APPLICATION_CREDENTIALS"

        if gcloud auth print-access-token >/dev/null 2>&1; then
            ACTIVE_ACCOUNT=$(gcloud config get-value account 2>/dev/null || true)
            if [ -z "$ACTIVE_ACCOUNT" ] || [ "$ACTIVE_ACCOUNT" = "(unset)" ]; then
                ACTIVE_ACCOUNT=$(jq -r '.account // empty' "$GOOGLE_APPLICATION_CREDENTIALS")
            fi

            if [ -n "$ACTIVE_ACCOUNT" ] && [ "$ACTIVE_ACCOUNT" != "(unset)" ]; then
                echo "Using gcloud account: $ACTIVE_ACCOUNT"
            else
                echo "Using gcloud credential override from $GOOGLE_APPLICATION_CREDENTIALS"
            fi
        else
            echo "Warning: credential file override token check failed. Falling back to mounted gcloud config."
            unset CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE

            if [ -n "$GCP_ACCOUNT" ]; then
                gcloud config set account "$GCP_ACCOUNT" >/dev/null 2>&1 || true
            fi

            ACTIVE_ACCOUNT=$(gcloud config get-value account 2>/dev/null || true)
            if [ -z "$ACTIVE_ACCOUNT" ] || [ "$ACTIVE_ACCOUNT" = "(unset)" ]; then
                echo "Error: no active gcloud account available. Run 'gcloud auth login' on host and mount ~/.config/gcloud, or use a service account key."
                exit 1
            fi
            echo "Using gcloud account: $ACTIVE_ACCOUNT"
        fi
    else
        echo "Warning: unsupported credential file type '$CRED_TYPE'. Falling back to mounted gcloud config."
    fi
else
    echo "Note: Assuming gcloud config is mounted from host."
fi

if [ -n "$PROJECT_ID" ]; then
    gcloud config set project "$PROJECT_ID"
    echo "Set project to $PROJECT_ID"
fi

# Run the watchdogs in the background using bash instead of tmux
# Tmux needs a TTY to attach properly which we might not have in docker
# Stagger startups by 15s to prevent simultaneous TPU creation quota hits
CONFIG_PATTERN=${WATCHDOG_CONFIG_PATTERN:-"*.conf"}
shopt -s nullglob
CONFIGS=(/app/tpu_orchestration/configs/$CONFIG_PATTERN)

if [ ${#CONFIGS[@]} -eq 0 ]; then
    echo "Error: no watchdog configs matched pattern '$CONFIG_PATTERN'."
    exit 1
fi

echo "Using watchdog config pattern: $CONFIG_PATTERN"
DELAY=0
for conf in "${CONFIGS[@]}"; do
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
