#!/usr/bin/env sh
# Executed on each TPU pod worker via `gcloud tpu-vm scp` + `gcloud tpu-vm ssh --worker=all`.
# Authenticates with Artifact Registry using the VM's service account metadata token,
# pulls the TPU trainer image, then runs the W&B sweep agent inside Docker.
# TPU chip devices (/dev/accel*) are exposed via --privileged + /dev volume mount.
# Required env vars: WANDB_API_KEY, SWEEP_ID
# Optional: AGENT_COUNT (default 1, 0 = run until sweep ends)
set -eu

IMAGE="us-central1-docker.pkg.dev/phantom-trc/phantom/phantom-trainer:tpu-latest"
AGENT_COUNT="${AGENT_COUNT:-1}"

# use VM service account — no manual key needed on the pod
TOKEN=$(curl -sf -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" \
  | python3 -c 'import sys, json; print(json.load(sys.stdin)["access_token"])')

echo "$TOKEN" | sudo docker login -u oauth2accesstoken \
  --password-stdin https://us-central1-docker.pkg.dev

sudo docker pull "$IMAGE"

# --privileged + /dev mount gives the container access to /dev/accel* (TPU chips)
# --network host lets JAX reach the other pod workers for distributed init
sudo docker run --rm \
  --privileged \
  --network host \
  --volume /dev:/dev \
  -e WANDB_API_KEY="$WANDB_API_KEY" \
  -e SWEEP_ID="$SWEEP_ID" \
  -e AGENT_COUNT="$AGENT_COUNT" \
  "$IMAGE"
