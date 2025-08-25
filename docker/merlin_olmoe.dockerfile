# FROM hub.byted.org/base/lab.pytorch2:dcbfbcec1ef8c7885d539adee6c3b62c

# ARG https_proxy="http://sys-proxy-rd-relay.byted.org:8118"
# ARG http_proxy="http://sys-proxy-rd-relay.byted.org:8118"
# ARG no_proxy="byted.org"

# # 安装常用工具，conda
# RUN pip3 install --no-cache-dir deepspeed==0.14.1  &&  \
#     pip3 install --no-cache-dir transformers==4.40.0  &&  \
#     pip3 install --no-cache-dir accelerate==0.29.3  &&  \
#     pip3 install --no-cache-dir xformers==0.0.25.post1  &&  \
#     pip3 install --no-cache-dir sentencepiece==0.2.0  &&  \
#     pip3 install --no-cache-dir peft==0.10.0 && \
#     pip3 install --no-cache-dir byted-cruise -i "https://bytedpypi.byted.org/simple" && \
#     pip3 install --no-cache-dir timm==0.9.16 && \
#     pip3 install --no-cache-dir diffusers==0.27.2 && \
#     pip3 install --no-cache-dir accelerate==0.29.3 && \
#     pip3 install --no-cache-dir pytorch-ignite==0.5.0.post2 && \
#     pip3 install --no-cache-dir pytorch-fid==0.3.0 && \
#     pip3 install --no-cache-dir -U byted-wandb -i https://bytedpypi.byted.org/simple && \
#     pip3 install --no-cache-dir packaging ninja && \
#     pip3 install --no-cache-dir flash-attn --no-build-isolation --no-cache && \
#     pip3 install --no-cache-dir decord && \
#     pip3 install --no-cache-dir scipy && \
#     pip3 install --no-cache-dir moviepy imageio && \
#     pip3 install --no-cache-dir vector_quantize_pytorch && \
#     pip3 install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex.git && \
#     pip3 install --no-cache-dir -U xformers --index-url https://download.pytorch.org/whl/cu121  && \
#     pip3 install --no-cache-dir tensorflow-cpu  && \
#     pip3 install --no-cache-dir byted-dataloader==0.4.4 && \
#     pip3 install --no-cache-dir colorama && \
#     pip3 install --no-cache-dir jsonpath-ng && \
#     pip3 install --no-cache-dir opencv-python && \
#     pip3 install --no-cache-dir redis && \
#     pip3 install --no-cache-dir cached-path && \
#     pip3 install --no-cache-dir omegaconf && \
#     pip3 install --no-cache-dir torchmetrics && \
#     pip3 install --no-cache-dir datasets && \
#     pip3 install --no-cache-dir scikit-learn && \
#     pip3 uninstall -y flash-attn && \
#     pip3 install flash-attn --no-build-isolation && \
#     pip3 install ftfy && \
#     pip3 uninstall -y apex && \
#     pip3 install pillow pillow-avif-plugin pillow-heif && \
#     pip3 install rotary_embedding_torch && \
#     apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# hub.byted.org/base/seed_pytorch2_python39:5f66d37d4c2e9021c5d8c5c37823240a 来自上面的dockerfile
ARG REGION

FROM hub.byted.org/base/seed_pytorch2_python39:5f66d37d4c2e9021c5d8c5c37823240a as china-north-lf
FROM hub.byted.org/base/seed_pytorch2_python39:5f66d37d4c2e9021c5d8c5c37823240a as us-east
FROM hub.byted.org/base/seed_pytorch2_python39:5f66d37d4c2e9021c5d8c5c37823240a as us-east-red
FROM hub.byted.org/base/seed_pytorch2_python39:5f66d37d4c2e9021c5d8c5c37823240a as aliyun_sg
FROM hub.byted.org/base/seed_pytorch2_python39:5f66d37d4c2e9021c5d8c5c37823240a as aliyun_va

ENV http_proxy=http://sys-proxy-rd-relay.byted.org:3128
ENV https_proxy=http://sys-proxy-rd-relay.byted.org:3128
ENV no_proxy=*.byted.org

FROM ${REGION} as devel

RUN pip3 install --no-cache-dir numpy && \
    pip3 install --no-cache-dir "torch>=2.1,<2.4" && \
    pip3 install --no-cache-dir "ai2-olmo-core==0.1.0" && \
    pip3 install --no-cache-dir omegaconf && \
    pip3 install --no-cache-dir rich && \
    pip3 install --no-cache-dir boto3 && \
    pip3 install --no-cache-dir google-cloud-storage && \
    pip3 install --no-cache-dir tokenizers && \
    pip3 install --no-cache-dir packaging && \
    pip3 install --no-cache-dir "cached_path>=1.6.2" && \
    pip3 install --no-cache-dir transformers && \
    pip3 install --no-cache-dir importlib_resources && \
    pip3 install git+https://github.com/Muennighoff/megablocks.git@4a25bc7b5665bcb9da93d72d5ad0c14d41e1a351 && \
    pip3 install git+https://github.com/Muennighoff/megablocks.git@e430ad707bed4d45016f315da9372e16acb55a1c
