#!/bin/bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
third_party_dir="${script_dir}/third_party"

mkdir -p "${third_party_dir}"
cd "${third_party_dir}"

URL_BASE="https://gitee.com/LockzhinerAI"

# 定义需要下载的文件名列表
FILES=(
    "opencv-mobile/releases/download/v29/opencv-mobile-4.10.0-lockzhiner-vision-module.zip"
    "fmt/releases/download/11.0.2/fmt-11.0.2-lockzhiner-vision-module.zip"
    "pybind11/releases/download/v2.13.5/pybind11-v2.13.5.zip"
    "pybind11/releases/download/v2.13.5/python3.11-lockzhiner-vision-module.zip"
    "zxing-cpp/releases/download/v2.2.1/zxing-cpp-v2.2.1-lockzhiner-vision-module.zip"
    "ncnn/releases/download/20240820/ncnn-20240820-lockzhiner-vision-module.zip"
    "apriltag-with-pose-estimation-master/releases/download/apriltag-with-pose-estimation-master/apriltag-with-pose-estimation-master.zip"
    "eigen-master/releases/download/v0.1/eigen.zip"
    "jsoncpp/releases/download/v0.1/jsoncpp.zip"
    "opus/releases/download/v0.1/opus-v1.5.2.zip"
    "LockzhinerVisionModule/releases/download/v0.0.6/lockzhiner_vision_module_sdk.zip"
)

echo "--- 开始一键下载并解压所有依赖库 ---"

if ! command -v wget >/dev/null 2>&1; then
    echo "缺少 wget，请先安装后再运行。"
    exit 1
fi
if ! command -v unzip >/dev/null 2>&1; then
    echo "缺少 unzip，请先安装后再运行。"
    exit 1
fi

for path in "${FILES[@]}"; do
    filename=$(basename "$path")
    if [ ! -f "$filename" ]; then
        echo "正在下载: $filename ..."
        wget "${URL_BASE}/${path}" -O "$filename"
    fi
    echo "正在解压: $filename ..."
    unzip -qo "$filename"
done

toolchain_dir="${script_dir}/toolchain"
toolchain_marker="${toolchain_dir}/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc"
if [ ! -x "${toolchain_marker}" ]; then
    if [ -e "${toolchain_dir}" ]; then
        echo "检测到 ${toolchain_dir} 已存在但工具链不完整。"
        echo "请删除该目录后重新运行：rm -rf \"${toolchain_dir}\""
        exit 1
    fi

    if ! command -v git >/dev/null 2>&1; then
        echo "缺少 git，请先安装后再运行。"
        exit 1
    fi

    echo "正在下载: arm-rockchip830-linux-uclibcgnueabihf 工具链 ..."
    git clone --depth 1 https://gitee.com/LockzhinerAI/arm-rockchip830-linux-uclibcgnueabihf.git "${toolchain_dir}"
fi

echo "--- 全部搞定！你的 third_party 文件夹已就绪 ---"
echo "接下来你可以运行以下命令启动开发环境："
echo "docker compose up -d"
echo "docker exec -it rv1106_dev bash"
