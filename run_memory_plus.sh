pkill -9 -f redis-server
pkill -9 -f wandb-service
pkill -9 -f train

proxy()
{
    export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
    export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
    export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
    export no_proxy="byted.org,bytedance.net,.byted.org,.bytedance.net,localhost,127.0.0.1,::1,10.0.0.0/8,127.0.0.0/8,fd00::/8,100.64.0.0/10,fe80::/10,172.16.0.0/12,169.254.0.0/16,192.168.0.0/16"
}
unproxy()
{
    unset HTTP_PROXY
    unset http_proxy
    unset https_proxy
}

CUR_DEVICE_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader)
CUR_DEVICE_TYPE=$(echo "${CUR_DEVICE_TYPE}" | tr '[:upper:]' '[:lower:]')
if echo "${CUR_DEVICE_TYPE}" | grep -q -e "h100" -e "h800"; then
    echo "Using NVIDIA H-Series GPU setting"
    DEVICE_SERIES="H"
    PKG_SUFFIX=".h800"
else
    echo "Using NVIDIA A-Series GPU setting"
    DEVICE_SERIES="A"
    PKG_SUFFIX=""
fi

set -ex
export HF_DATASETS_OFFLINE=1
# =========================== 重要配置 ===========================
MOUNT_DIR="/mnt/hdfs/yutao_ssd_hldy"
CODE_DIR="/opt/tiger/olmoe"

# 以下为可修改的几个配置
export run_name="debug_hzh_memory_plus"
CONFIG_PATH=${CODE_DIR}/configs/exps/OLMoE-1B-7B-0906_reproduce.yml

SAVE_DIR="/mnt/hdfs/hhz_ssd_hldy/olmoe_exps/${run_name}"
# =========================== 重要配置 ===========================

# =========================== 环境初始化(无需改动) ===========================
if [ -d /mnt/bn/mount_nas ]; then
    echo "/mnt/bn/mount_nas exists, skip ..."
else
    sudo mkdir -p /mnt/bn/mount_nas
    sudo ln -s ${MOUNT_DIR}/corpus /mnt/bn/mount_nas/datasets
fi
echo "MOUNT_DIR = ${MOUNT_DIR}"
echo "CODE_DIR = ${CODE_DIR}"


OLMO_ENV_INIT_FLAG_FILE="${CODE_DIR}/OLMO_ENV_INITIALIZED.lock"
if [ -e "${OLMO_ENV_INIT_FLAG_FILE}" ]; then
    echo "olmo env initialized, skip ... (for hotfix)"
else
    if echo "${ARNOLD_REGION}" | grep -q -e "CN"; then
        proxy
        echo "Using proxy"
    fi
    # 这两个包的安装顺序需要保证: 先olmoe -> 后zloss
    pip3 install /mnt/bn/mount_nas/datasets/olmoe_related/pkgs/megablocks-0.5.1+olmoe${PKG_SUFFIX}-cp39-cp39-linux_x86_64.whl
    pip3 install /mnt/bn/mount_nas/datasets/olmoe_related/pkgs/megablocks-0.5.1+zloss${PKG_SUFFIX}-cp39-cp39-linux_x86_64.whl
    # pip3 install git+https://github.com/Muennighoff/megablocks.git@4a25bc7b5665bcb9da93d72d5ad0c14d41e1a351 && \
    # pip3 install git+https://github.com/Muennighoff/megablocks.git@e430ad707bed4d45016f315da9372e16acb55a1c
    # pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
    if echo "${ARNOLD_REGION}" | grep -q -e "CN"; then
        unproxy
        echo "Unset proxy"
    fi
    mkdir -p ~/.cache
    pushd ~/.cache
    tar --keep-newer-files -xzf /mnt/bn/mount_nas/datasets/olmoe_related/huggingface_cache_v3.tar.gz
    popd
    touch ${OLMO_ENV_INIT_FLAG_FILE}
fi
# =========================== 环境初始化(无需改动) ===========================

# =========================== 运行脚本 ===========================
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 以下为自动resume的配置脚本, 如果需要中途修改load_path, 需要去修改hdfs目录下的latest_checkpointed_iteration.txt文件
if [ -e ${SAVE_DIR}/latest_checkpointed_iteration.txt ]; then
    read -r CUR_STEP < ${SAVE_DIR}/latest_checkpointed_iteration.txt
    echo "current latest_checkpointed_iteration = ${CUR_STEP}"
    if [ -d ${CODE_DIR}/${CUR_STEP} ]; then
        echo "${CODE_DIR}/${CUR_STEP} already exists, skip downloading ..."
    else
        cp -r /mnt/hdfs/hhz_ssd_hldy/olmoe_exps/${run_name}/${CUR_STEP} ${CODE_DIR}/
    fi
    CUR_CKPT_PATH="${CODE_DIR}/${CUR_STEP}"
else
    CUR_CKPT_PATH="auto"
fi

echo "trial_load_path = ${CUR_CKPT_PATH}"

# 以下为hotfix常用配置
# CUR_BRANCH="zhudefa/olmoe_hc_hdfs"
# CUR_COMMIT="22ce4b63b3c4cafdea0a84327507f6fe6beb3a36"
# git stash && git fetch origin ${CUR_BRANCH} && git checkout ${CUR_BRANCH} && git reset --hard ${CUR_COMMIT}

sh launch.sh ${CONFIG_PATH} \
--save_folder=${SAVE_DIR} \
--run_name=${run_name} \
--save_overwrite=true \
--mount_common_hdfs=true \
--fsdp.sharding_strategy=NO_SHARD \
--canceled_check_interval=9999999 \
--global_indices_file=${CODE_DIR}/global_indices.npy \
--load_path=${CUR_CKPT_PATH} \
--model.init_std=0.02282177322938192 \
--model.init_fn="full_megatron" \
--model.d_model=768 \
--model.n_layers=20 \
--model.n_heads=12 \
--model.n_kv_heads=12 \
--model.weight_tying=true \
--max_duration=5e11T \
--scheduler.t_warmup=1e10 \
--scheduler.t_max=5e11 \
--device_train_microbatch_size=2 \
--global_train_batch_size=16 \
--save_interval=1000 \
--eval_interval=1000 \
--save_num_checkpoints_to_keep=-1 \
--console_log_interval=1 \
\
--model.block_type='sequential' \
--model.mlp_hidden_size=6016 \
\
--optimizer.mem_value_lr_times=4.0 \
--optimizer.mem_value_lr_max_steps_rate=1.0 \
--model.mem_insert_way='4:4/8:8/12:12/16:16' \
--model.mem_knum=1138 \
--model.mem_kdim=384 \
--model.mem_vdim=768 \
--model.mem_knn=80 \
--model.mem_head=2 \
--model.mem_type='memory_plus' \
\
--distributed_strategy=ddp \
--model.init_device='cuda' \
--optimizer.metrics_log_interval=10 \
--model.mem_log_interval=10 \
--save_interval_unsharded=1000 \
--ddp.grad_sync_mode=micro_batch \
--ddp.find_unused_params=true
