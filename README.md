# RV1106 视觉模块快速开发环境 (Lockzhiner Vision Module)

## 📢 来源声明

本项目中的所有例程（C++ Examples）均来源于凌智电子（Lockzhiner）开源的官方仓库：
👉 [LockzhinerVisionModule](https://gitee.com/LockzhinerAI/LockzhinerVisionModule)

本仓库在官方例程的基础上，**提供了一个经过优化的 Docker 开发环境**。通过引入国内镜像加速和自动化脚本，帮助开发者快速搭建出开箱即用的 RV1106 交叉编译环境。

---

## 🚀 快速开发环境搭建教程

### 前置条件
- 已安装 [Git](https://git-scm.com/)
- 已安装 [Docker](https://www.docker.com/) 和 [Docker Compose](https://docs.docker.com/compose/)

### 1. 初始化依赖库
例程依赖了 OpenCV、NCNN、Eigen 等一系列第三方库。我们提供了一个自动化脚本，一键拉取所有必需的库到 `third_party` 目录下。

```bash
chmod +x setup_rv1106.sh
./setup_rv1106.sh
```

### 2. 构建并启动环境
在项目根目录下执行以下命令，Docker 会自动构建镜像并启动容器。

```bash
# 构建并后台运行容器
docker compose up -d --build
```

### 3. 进入开发环境
容器启动后，当前目录会自动挂载到容器内的 `/home/rv1106/workspace`。

```bash
# 进入交互式终端
docker exec -it rv1106_dev bash
```
进入容器后，默认用户为 `rv1106`，已配置好交叉编译工具链环境变量。

### 4. 编译 C++ 例程
进入容器后，进入例程目录并使用 CMake 进行编译。推荐使用以下现代 CMake 命令：

```bash
# 以 A01_helloworld 为例
cd Cpp_example/A01_helloworld

# 配置 (生成 build 目录并指定工具链)
cmake -B build -DCMAKE_TOOLCHAIN_FILE=../../toolchains/arm-rockchip830-linux-uclibcgnueabihf.toolchain.cmake

# 编译 (多线程并行)
cmake --build build --parallel
```

---

## 💡 容器管理常用命令

| 操作 | 命令 |
| :--- | :--- |
| **查看容器状态** | `docker ps --filter "name=rv1106_dev"` |
| **停止容器** | `docker compose stop` |
| **启动容器** | `docker compose start` |
| **移除环境** | `docker compose down` |

## 🔗 参考资料
- [凌智视觉模块官方 Gitee 仓库](https://gitee.com/LockzhinerAI/LockzhinerVisionModule)
- [凌智视觉模块 Bilibili 视频专栏](https://space.bilibili.com/3546390319842030)
