FROM lcp_pinn-dataset:v2

ARG USERNAME=teleophost
ARG USER_UID=1000
ARG USER_GID=1000

USER root

RUN apt-get update && \
    apt-get install -y \
        git \
        libegl1 \
        libgl1 \
        libglfw3 \
        libglib2.0-0 \
        libosmesa6 \
        libx11-6 \
        libxcursor1 \
        libxext6 \
        libxinerama1 \
        libxi6 \
        libxrandr2 \
        python3-tk && \
    rm -rf /var/lib/apt/lists/*

RUN /opt/venv/bin/python -m pip install --no-cache-dir \
    pyrealsense2 \
    opencv-python \
    numpy \
    "huggingface-hub[cli,hf-transfer]>=0.34.2,<0.36.0" \
    "fsspec[http]>=2023.1.0,<=2025.3.0" \
    "packaging>=24.2,<26.0" \
    ultralytics \
    open3d \
    git+https://github.com/ultralytics/CLIP.git

RUN /opt/venv/bin/python -m pip install --no-cache-dir \
    "mujoco>=3.1.3,<3.4.0" \
    "cmeel-tinyxml2==10.0.0" \
    "cmeel-urdfdom==4.0.1" \
    "libpinocchio==3.9.0" \
    "pin==3.9.0"

# dp_hirol policy inference extras. Torch/torchvision stay inherited from the
# base LCP-PINN image so their CUDA ABI remains aligned with this repository.
RUN /opt/venv/bin/python -m pip install --no-cache-dir \
    "hydra-core==1.3.2" \
    "omegaconf==2.3.0" \
    "dill==0.3.8" \
    "diffusers==0.35.1" \
    "transformers==4.56.2" \
    "einops==0.8.1" \
    "pandas==2.2.3" \
    "pyarrow==18.1.0" \
    "av==15.0.0" \
    "zarr==2.18.3" \
    "numcodecs==0.12.1" \
    "scipy==1.11.4" \
    "tqdm==4.67.1"

# 准备一个和宿主机 UID/GID 对齐的运行环境。
# 基础镜像里可能已经有 UID=1000 的用户（例如 trainer），这种情况下不再强行创建同 UID 用户。
RUN set -eux; \
    if ! getent group "${USER_GID}" >/dev/null; then \
        if getent group "${USERNAME}" >/dev/null; then \
            groupmod --gid "${USER_GID}" "${USERNAME}"; \
        else \
            groupadd --gid "${USER_GID}" "${USERNAME}"; \
        fi; \
    fi; \
    if id "${USERNAME}" >/dev/null 2>&1; then \
        usermod --uid "${USER_UID}" --gid "${USER_GID}" --home "/home/${USERNAME}" "${USERNAME}"; \
    elif getent passwd "${USER_UID}" >/dev/null; then \
        true; \
    else \
        useradd --uid "${USER_UID}" --gid "${USER_GID}" --create-home --shell /bin/bash "${USERNAME}"; \
    fi; \
    mkdir -p "/home/${USERNAME}" /workspace /tmp/matplotlib; \
    chmod 1777 /tmp/matplotlib; \
    chown -R "${USER_UID}:${USER_GID}" "/home/${USERNAME}" /workspace

ENV HOME=/home/${USERNAME}
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV CMAKE_PREFIX_PATH=/opt/venv/lib/python3.10/site-packages/cmeel.prefix
ENV LD_LIBRARY_PATH=/opt/venv/lib/python3.10/site-packages/cmeel.prefix/lib
WORKDIR /workspace

USER ${USER_UID}:${USER_GID}
