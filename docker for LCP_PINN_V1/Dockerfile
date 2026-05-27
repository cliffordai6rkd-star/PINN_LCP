FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG USERNAME=teleophost
ARG USER_UID=1000
ARG USER_GID=1000

ARG PYTORCH_VERSION=2.6.0
ARG TORCHVISION_VERSION=0.21.0
ARG TORCHAUDIO_VERSION=2.6.0
ARG TORCHCODEC_VERSION=0.2.1
ARG TORCH_CUDA=cu124
ARG LEROBOT_VERSION=0.4.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display,video

USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        dbus \
        ffmpeg \
        git \
        libavcodec-dev \
        libavdevice-dev \
        libavfilter-dev \
        libavformat-dev \
        libavutil-dev \
        libegl1 \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libswresample-dev \
        libswscale-dev \
        libusb-1.0-0 \
        libx11-6 \
        libxcb1 \
        libxcursor1 \
        libxext6 \
        libxi6 \
        libxinerama1 \
        libxrandr2 \
        libxrender1 \
        pkg-config \
        python-is-python3 \
        python3 \
        python3-dev \
        python3-pip \
        python3-tk \
        python3-venv \
        sudo \
        tini \
        usbutils \
        vim \
        wget && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade \
        pip \
        "setuptools>=71,<81" \
        wheel && \
    python3 -m pip install \
        --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" \
        "torch==${PYTORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        "torchaudio==${TORCHAUDIO_VERSION}" \
        "torchcodec==${TORCHCODEC_VERSION}" && \
    python3 -m pip install \
        "lerobot==${LEROBOT_VERSION}" \
        h5py \
        matplotlib \
        pandas \
        pyarrow \
        pyyaml \
        tqdm && \
    python3 -c "import torch, torchvision, torchcodec, h5py, yaml; from lerobot.datasets.lerobot_dataset import LeRobotDataset; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('torchvision', torchvision.__version__)"

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
    mkdir -p "/home/${USERNAME}" \
        "/home/${USERNAME}/.cache" \
        "/home/${USERNAME}/.config" \
        /workspace; \
    chown -R "${USER_UID}:${USER_GID}" "/home/${USERNAME}" /workspace; \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >/etc/sudoers.d/"${USERNAME}"; \
    chmod 0440 /etc/sudoers.d/"${USERNAME}"

ENV HOME=/home/${USERNAME} \
    WORKDIR=/workspace \
    PYTHONPATH=/workspace

WORKDIR /workspace

USER ${USER_UID}:${USER_GID}

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]
