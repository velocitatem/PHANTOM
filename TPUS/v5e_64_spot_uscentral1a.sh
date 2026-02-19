# 64 spot Cloud TPU v5e chips in zone us-central1-a
export PROJECT_ID=phantom-trc
export QR_NAME=TPUv5e64spotUC1A
export TPU_NAME=tpu-v5e-64-uc1a
export ZONE=us-central1-a
export ACCELERATOR_TYPE=v5e-64
export RUNTIME_VERSION=v2-alpha-tpuv5-lite

gcloud compute tpus tpu-vm create ${TPU_NAME} \
       --project=${PROJECT_ID} \
       --zone=${ZONE} \
       --accelerator-type=${ACCELERATOR_TYPE} \
       --version=${RUNTIME_VERSION} \
       --spot \
|| \
gcloud compute tpus queued-resources create ${QR_NAME} \
       --project=${PROJECT_ID} \
       --zone=${ZONE} \
       --node-id=${TPU_NAME} \
       --accelerator-type=${ACCELERATOR_TYPE} \
       --runtime-version=${RUNTIME_VERSION} \
       --spot
