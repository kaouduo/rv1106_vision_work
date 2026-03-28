# jsoncpp 测试例程
## 简介

JSON 是一种轻量级数据交换格式。它可以表示数据、字符串、有序的值序列以及名称/值对的集合。

jsoncpp 是一个 C++ 库，允许操作 JSON 值，包括字符串之间的序列化和反序列化。它还可以在反序列化/序列化步骤中保留现有注释，使其成为存储用户输入文件的方便格式。

JsonCpp 目前在 github 上托管。为方便使用将改项目继承到 SDk 中。

官方网址：https://github.com/open-source-parsers/jsoncpp

## 编译项目

使用 Docker Destop 打开 LockzhinerVisionModule 容器并执行以下命令来编译项目

```bash
# 进入Demo所在目录
cd /LockzhinerVisionModuleWorkSpace/LockzhinerVisionModule/Cpp_example/E01_find_number
# 创建编译目录
rm -rf build && mkdir build && cd build
# 配置交叉编译工具链
export TOOLCHAIN_ROOT_PATH="/LockzhinerVisionModuleWorkSpace/arm-rockchip830-linux-uclibcgnueabihf"
# 使用cmake配置项目
cmake ..
# 执行编译项目
make -j8 && make install
```

在执行完上述命令后，会在build目录下生成可执行文件。
拷贝到视觉模块上运行：
```bash
chmod 777 ./jsontest
./jsontest + 具体的任务
```