# 64 spot Cloud TPU v5e chips in zone europe-west4-b
export PROJECT_ID=phantom-trc
export QR_NAME=TPUv5e64spotEW4B
export TPU_NAME=tpu-v5e-64-ew4b
export ZONE=europe-west4-b
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
