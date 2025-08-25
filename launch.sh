set -x
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:=0}
export NCCL_IB_HCA=${NCCL_IB_HCA:-"$ARNOLD_RDMA_DEVICE:1"}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:=3}
# export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:=eth0}
# export NCCL_SOCKET_IFNAME="en,eth,em,bond"

# 兼容debug和正式任务中不同的role
ROLE="EXECUTOR"
if [ -z "$ARNOLD_EXECUTOR_0_HOST" ]; then
    ROLE="WORKER"
fi

master_addr=$(eval echo \${ARNOLD_${ROLE}_0_HOST})
master_port=$(eval echo \${ARNOLD_${ROLE}_0_PORT})
nnodes=$(eval echo \${ARNOLD_${ROLE}_NUM})

export NPROC=$(eval echo \${ARNOLD_${ROLE}_GPU})
export OMP_NUM_THREADS=4

torchrun ${ADDITIONAL_TORCHRUN_ARGS} \
    --nproc_per_node=$NPROC \
    --master_addr=${master_addr} \
    --master_port=${master_port} \
    --node_rank=$ARNOLD_ID \
    --nnodes=${nnodes} \
    scripts/train.py $@
