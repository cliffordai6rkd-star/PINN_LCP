FROM lcp_pinn-dataset:v2

ARG USERNAME=teleophost
ARG USER_UID=1000
ARG USER_GID=1000

USER root

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
    mkdir -p "/home/${USERNAME}" /workspace; \
    chown -R "${USER_UID}:${USER_GID}" "/home/${USERNAME}" /workspace

ENV HOME=/home/${USERNAME}
WORKDIR /workspace

USER ${USER_UID}:${USER_GID}
