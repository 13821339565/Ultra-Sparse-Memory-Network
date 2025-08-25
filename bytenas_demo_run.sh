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

set -ex
export HF_DATASETS_OFFLINE=1
# =========================== 重要配置 ===========================
export run_name="tmp_run"
MOUNT_DIR="/mnt/bn/lq-zengyutao-2"
CODE_DIR=$(pwd)
CONFIG_PATH=${CODE_DIR}/configs/exps/OLMoE-1B-7B-0906_reproduce.yml
SAVE_DIR="${MOUNT_DIR}/exps/${run_name}"
# =========================== 重要配置 ===========================

if [ -d /mnt/bn/mount_nas ]; then
    echo "/mnt/bn/mount_nas exists, skip ..."
else
    sudo ln -s ${MOUNT_DIR} /mnt/bn/mount_nas
fi
echo "MOUNT_DIR = ${MOUNT_DIR}"
echo "CODE_DIR = ${CODE_DIR}"


OLMO_ENV_INIT_FLAG_FILE="/opt/tiger/olmoe/OLMO_ENV_INITIALIZED.lock"
if [ -e "${OLMO_ENV_INIT_FLAG_FILE}" ]; then
    echo "olmo env initialized, skip ... (for hotfix)"
else
    proxy
    hdfs dfs -get hdfs://haruna/home/byte_data_seed/ssd_hldy/user/yutao.zeng/other/pkgs/megablocks-0.5.1+olmoe-cp39-cp39-linux_x86_64.whl
    hdfs dfs -get hdfs://haruna/home/byte_data_seed/ssd_hldy/user/yutao.zeng/other/pkgs/megablocks-0.5.1+zloss-cp39-cp39-linux_x86_64.whl
    pip3 install megablocks-0.5.1+olmoe-cp39-cp39-linux_x86_64.whl && pip3 install megablocks-0.5.1+zloss-cp39-cp39-linux_x86_64.whl
    rm megablocks*.whl
    # pip3 install git+https://github.com/Muennighoff/megablocks.git@4a25bc7b5665bcb9da93d72d5ad0c14d41e1a351 && \
    # pip3 install git+https://github.com/Muennighoff/megablocks.git@e430ad707bed4d45016f315da9372e16acb55a1c
    unproxy
    mkdir -p ~/.cache
    pushd ~/.cache
    tar --keep-newer-files -xzf ${MOUNT_DIR}/datasets/olmoe_related/huggingface_cache_v3.tar.gz
    popd
    touch ${OLMO_ENV_INIT_FLAG_FILE}
fi

# =========================== 开始运行 ===========================
sh launch.sh ${CONFIG_PATH} \
--save_folder=${SAVE_DIR} \
--run_name=${run_name} \
--save_overwrite=true \
--fsdp.sharding_strategy=FULL_SHARD \
--device_train_microbatch_size=2 \
--canceled_check_interval=9999999 \
--load_path=auto \
--save_interval=50 \
--eval_interval=150 \
--global_indices_file=${CODE_DIR}/global_indices.npy \
--save_num_checkpoints_to_keep=2 \
--global_train_batch_size=32 \
2>&1 | tee run_${ARNOLD_ID}.log
# =========================== 开始运行 ===========================