# 64 spot Cloud TPU v6e chips in zone europe-west4-a
export PROJECT_ID=phantom-trc
export QR_NAME=TPUv6e64spotEW4A
export TPU_NAME=tpu-v6e-64-ew4a
export ZONE=europe-west4-a
export ACCELERATOR_TYPE=v6e-64
export RUNTIME_VERSION=v2-alpha-tpuv6e

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
