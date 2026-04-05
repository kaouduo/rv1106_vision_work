# 使用华为云镜像加速拉取 Ubuntu 22.04
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/ubuntu:22.04

# 避免在安装过程中出现交互式提示
ENV DEBIAN_FRONTEND=noninteractive

# 1. 配置 APT 镜像源 (中科大) 提高下载速度
RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.ustc.edu.cn@g' /etc/apt/sources.list \
    && sed -i 's/security.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list

# 2. 安装 RV1106 视觉开发所需的依赖项
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    ninja-build \
    bash-completion \
    ca-certificates \
    python3 \
    libncurses5-dev \
    vim \
    curl \
    sudo \
    bc \
    python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

# 3. 自动将 dash 切换为 bash (解决脚本兼容性问题)
RUN echo "dash dash/sh boolean false" | debconf-set-selections \
    && dpkg-reconfigure -f noninteractive dash

# 4. 创建非 root 用户 'rv1106' 并设置密码
RUN useradd -m -s /bin/bash rv1106 && \
    echo "rv1106:rv1106" | chpasswd && \
    adduser rv1106 sudo

# 5. 配置 sudoers，允许 'rv1106' 用户在不需要密码的情况下使用 sudo
RUN echo "rv1106 ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# 6. 设置工作目录
WORKDIR /home/rv1106/workspace

# 7. 预下载交叉编译工具链 (放 /opt 目录下)
RUN git clone https://gitee.com/LockzhinerAI/arm-rockchip830-linux-uclibcgnueabihf.git /opt/toolchain && \
    chown -R rv1106:rv1106 /opt/toolchain

# 8. 配置环境变量
ENV TOOLCHAIN_ROOT_PATH="/opt/toolchain"
ENV PATH="/opt/toolchain/bin:${PATH}"

# 9. 开启终端色彩和命令补全 (针对 rv1106 用户)
RUN echo "source /etc/bash_completion" >> /home/rv1106/.bashrc \
    && echo "export PS1='\[\e[32m\]\u@rv1106-dev\[\e[m\]:\[\e[34m\]\w\[\e[m\]\$ '" >> /home/rv1106/.bashrc \
    && chown rv1106:rv1106 /home/rv1106/.bashrc

# 切换到 'rv1106' 用户
USER rv1106

# 默认启动 bash
CMD ["/bin/bash"]
