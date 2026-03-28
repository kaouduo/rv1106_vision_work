# 使用 Ubuntu 22.04 作为基底
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/ubuntu:22.04

# 1. 配置镜像源 (中科大) 提高下载速度
RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.ustc.edu.cn@g' /etc/apt/sources.list

# 2. 安装必要的基础工具，避免后续手动 apt install
# 使用 DEBIAN_FRONTEND=noninteractive 避免安装过程中的交互弹窗
ENV DEBIAN_FRONTEND=noninteractive
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
    && rm -rf /var/lib/apt/lists/*

# 3. 自动将 dash 切换为 bash (对应你手册中的 dpkg-reconfigure dash)
RUN echo "dash dash/sh boolean false" | debconf-set-selections \
    && dpkg-reconfigure -f noninteractive dash

# 4. 设置工作目录
WORKDIR /workspace

# 5. 预下载交叉编译工具链 (这一步最耗时，放镜像里以后就不用等了)
# 注意：如果 Gitee 速度慢，可以考虑手动下载压缩包后用 COPY 指令放入
RUN git clone https://gitee.com/LockzhinerAI/arm-rockchip830-linux-uclibcgnueabihf.git /opt/toolchain

# 6. 配置环境变量
# 这样你进入容器后，TOOLCHAIN_ROOT_PATH 已经是设好的，直接能用
ENV TOOLCHAIN_ROOT_PATH="/opt/toolchain"
ENV PATH="/opt/toolchain/bin:${PATH}"

# 7. 开启终端色彩和命令补全
RUN echo "source /etc/bash_completion" >> ~/.bashrc \
    && echo "export PS1='\[\e[32m\]\u@rk1106-dev\[\e[m\]:\[\e[34m\]\w\[\e[m\]\$ '" >> ~/.bashrc

CMD ["/bin/bash"]