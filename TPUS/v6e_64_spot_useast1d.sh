# 64 spot Cloud TPU v6e chips in zone us-east1-d
export PROJECT_ID=phantom-trc
export QR_NAME=TPUv6e64spotUE1D
export TPU_NAME=tpu-v6e-64-ue1d
export ZONE=us-east1-d
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
