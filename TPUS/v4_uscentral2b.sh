# 32 on-demand Cloud TPU v4 chips in zone us-central2-b
export PROJECT_ID=phantom-trc
export QR_NAME=TPUlong
export ZONE=us-central2-b
export ACCELERATOR_TYPE=v4-32
export RUNTIME_VERSION=v2-alpha-tpuv4
#gcloud compute tpus tpu-vm create ${TPU_NAME}     --zone=${ZONE}     --project=${PROJECT_ID}     --accelerator-type=${ACCELERATOR_TYPE}     --version=${RUNTIME_VERSION}
gcloud compute tpus queued-resources create ${QR_NAME} \
       --project=${PROJECT_ID} \
       --zone=${ZONE} \
       --node-id=${TPU_NAME} \
       --accelerator-type=${ACCELERATOR_TYPE} \
       --runtime-version=${RUNTIME_VERSION}
