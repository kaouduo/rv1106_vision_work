# RV1106 视觉模块快速开发环境 (Lockzhiner Vision Module)

## 📢 来源声明

本项目中的所有例程（C++ Examples）均来源于凌智电子（Lockzhiner）开源的官方仓库：
👉 [LockzhinerVisionModule](https://gitee.com/LockzhinerAI/LockzhinerVisionModule)

本仓库在官方例程的基础上，**主要提供了一个极其方便的开发环境搭建教程**。通过引入 Docker 和一键下载脚本，旨在帮助开发者跳过繁琐的环境配置步骤，快速在 Windows/Linux/macOS 上搭建出开箱即用的 RV1106 交叉编译环境。

---

## 🚀 快速开发环境搭建教程

为了保证编译环境的纯净和跨平台一致性，本仓库采用 **Docker + Docker Compose** 进行环境隔离与构建。容器内已经为您配置好了 `arm-rockchip830-linux-uclibcgnueabihf` 交叉编译工具链以及相关环境变量。

### 前置条件
- 已安装 [Git](https://git-scm.com/)
- 已安装 [Docker](https://www.docker.com/) 和 [Docker Compose](https://docs.docker.com/compose/)
- 确保网络畅通（需要从 Gitee 下载工具链和依赖库）

### 第一步：获取本仓库
```bash
git clone https://github.com/kaouduo/rv1106_vision_work.git
cd rv1106_vision_work
```

### 第二步：一键下载第三方依赖库
例程依赖了 OpenCV、NCNN、Eigen 等一系列第三方库。我们提供了一个自动化脚本，一键拉取所有必需的库到 `third_party` 目录下。

**Linux / macOS (或者在 WSL / Git Bash 下执行):**
```bash
chmod +x download_libs.sh
./download_libs.sh
```
*(注：下载完成后，当前目录下会生成一个 `third_party` 文件夹，包含所有解压好的库。)*

### 第三步：构建并启动 Docker 开发环境
在项目根目录（包含 `docker-compose.yml` 的目录）下执行以下命令，Docker 会自动拉取 Ubuntu 22.04 基础镜像，配置依赖，并拉取交叉编译工具链（该步骤视网络情况可能需要几分钟）。

```bash
# 构建镜像并后台运行容器
docker compose up -d --build
```

### 第四步：进入容器开始开发
容器启动后，当前代码目录会自动挂载到容器内的 `/workspace` 路径中，你在主机上修改的代码会实时同步到容器内。

```bash
# 进入开发容器的交互式终端
docker exec -it lockzhiner_dev bash
```
进入容器后，你将看到类似 `root@rk1106-dev:/workspace#` 的提示符。此时环境变量 `TOOLCHAIN_ROOT_PATH` 已自动配置好，可以直接进行交叉编译。

### 第五步：编译 C++ 例程
在容器内，进入你想要测试的例程目录（例如 `Cpp_example/A06_LCD`），使用 CMake 结合我们提供的 toolchain 文件进行编译：

```bash
cd Cpp_example/A06_LCD
mkdir build && cd build

# 使用提供的 toolchain 文件进行交叉编译
cmake -DCMAKE_TOOLCHAIN_FILE=../../toolchains/arm-rockchip830-linux-uclibcgnueabihf.toolchain.cmake ..

make -j4
```
编译成功后，生成的产物可以直接传输到 RV1106 开发板上运行。

---

## 💡 容器管理常用命令

- **退出容器**：在容器终端内输入 `exit` 或按 `Ctrl+D`。
- **停止开发环境**：`docker compose stop`
- **销毁开发环境**（不影响你的代码）：`docker compose down`
- **重新进入已启动的容器**：`docker exec -it lockzhiner_dev bash`

## 🔗 参考资料
- [凌智视觉模块官方 Gitee 仓库](https://gitee.com/LockzhinerAI/LockzhinerVisionModule)
- [凌智视觉模块 Bilibili 视频专栏](https://space.bilibili.com/3546390319842030)
