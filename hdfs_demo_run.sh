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

AREA="i18n"
if echo "${ARNOLD_REGION}" | grep -q -e "CN"; then
    AREA="cn"
fi

echo "current area = ${AREA}"

set -ex
export HF_DATASETS_OFFLINE=1
# =========================== 重要配置 ===========================
MOUNT_DIR="/mnt/hdfs/yutao_ssd_hldy"
CODE_DIR="/opt/tiger/olmoe"

# 以下为可修改的几个配置
export run_name="tmp_run"
# run_name: 任务名, DEBUG_FLAG: 是否为debug环境(0 or 1, 防止hdfs上产生太多无效目录, 正式任务记得关)
DEBUG_FLAG=1
FORCE_PROXY=0
CONFIG_PATH=${CODE_DIR}/configs/exps/OLMoE-1B-7B-0906_reproduce.yml
# =========================== 重要配置 ===========================

# =========================== 环境初始化(无需改动) ===========================
if [ "$DEBUG_FLAG" = "1" ]; then
    mkdir -p ${CODE_DIR}/olmoe_exps
    SAVE_DIR="${CODE_DIR}/olmoe_exps/${run_name}"
else
    SAVE_DIR="${MOUNT_DIR}/olmoe_exps/${run_name}"
fi

if [ -d /mnt/bn/mount_nas ]; then
    echo "/mnt/bn/mount_nas exists, skip ..."
else
    sudo mkdir -p /mnt/bn/mount_nas
    sudo ln -s ${MOUNT_DIR}/corpus /mnt/bn/mount_nas/datasets
fi

if [ "$AREA" = "cn" ]; then
    REMOTE_HDFS_DIR="hdfs://haruna/home/byte_data_seed/ssd_hldy/user/yutao.zeng"
else
    REMOTE_HDFS_DIR="hdfs://harunava/home/byte_arnold_va_mlsys/user/yutao.zeng"
fi

echo "MOUNT_DIR = ${MOUNT_DIR}"
echo "CODE_DIR = ${CODE_DIR}"
echo "REMOTE_HDFS_DIR = ${REMOTE_HDFS_DIR}"


OLMO_ENV_INIT_FLAG_FILE="/opt/tiger/olmoe/OLMO_ENV_INITIALIZED.lock"
if [ -e "${OLMO_ENV_INIT_FLAG_FILE}" ]; then
    echo "olmo env initialized, skip ... (for hotfix)"
else
    if [ "${AREA}" = "cn" ] || [ "${FORCE_PROXY}" = "1" ]; then
        proxy
        echo "Using proxy"
    fi
    # 这两个包的安装顺序需要保证: 先olmoe -> 后zloss
    pip3 install /mnt/bn/mount_nas/datasets/olmoe_related/pkgs/megablocks-0.5.1+olmoe${PKG_SUFFIX}-cp39-cp39-linux_x86_64.whl
    pip3 install /mnt/bn/mount_nas/datasets/olmoe_related/pkgs/megablocks-0.5.1+zloss${PKG_SUFFIX}-cp39-cp39-linux_x86_64.whl
    # pip3 install git+https://github.com/Muennighoff/megablocks.git@4a25bc7b5665bcb9da93d72d5ad0c14d41e1a351 && \
    # pip3 install git+https://github.com/Muennighoff/megablocks.git@e430ad707bed4d45016f315da9372e16acb55a1c
    if [ "${AREA}" = "cn" ] || [ "${FORCE_PROXY}" = "1" ]; then
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
        hdfs dfs -get -t 16 ${REMOTE_HDFS_DIR}/olmoe_exps/${run_name}/${CUR_STEP} ${CODE_DIR}/
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
--fsdp.sharding_strategy=FULL_SHARD \
--canceled_check_interval=9999999 \
--load_path=${CUR_CKPT_PATH} \
--global_indices_file=${CODE_DIR}/global_indices.npy \
--model.init_std=0.001976423538 \
--model.d_model=1024 \
--model.n_layers=32 \
--model.n_heads=16 \
--model.n_kv_heads=4 \
--max_duration=5e11T \
--scheduler.t_warmup=1e10 \
--device_train_microbatch_size=4 \
--global_train_batch_size=64 \
--save_interval=1000 \
--eval_interval=1000 \
--save_num_checkpoints_to_keep=40

# 如果显存开销太大, 可以设置--activation_checkpointing=fine_grained