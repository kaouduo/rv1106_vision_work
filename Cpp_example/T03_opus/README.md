# Opus 测试例程
## 简介

Opus是一款完全开放、免版税、高度通用的音频编解码器。Opus在互联网上的交互式语音和音乐传输方面无与伦比，但也适用于存储和流媒体应用。它被互联网工程任务组（IETF）标准化为 RFC 6716，它结合了Skype的SILK编解码器和Xiph.Org的CELT编解码器的技术。

Opus可以处理广泛的音频应用程序，包括IP语音、视频会议、游戏内聊天，甚至远程现场音乐表演。它可以从低比特率的窄带语音扩展到非常高质量的立体声音乐。支持的功能包括：

比特率从6 kb/s到510 kb/s

采样率从8kHz（窄带）到48kHz（全频带）

帧大小从2.5毫秒到60毫秒

支持恒定比特率（CBR）和可变比特率（VBR）

从窄带到全频带的音频带宽

支持语音和音乐

支持单声道和立体声

最多支持255个通道（多流帧）

可动态调整的比特率、音频带宽和帧大小

良好的丢失鲁棒性和丢包隐藏性（PLC）

浮点和定点实现

需要了解更多opus的内容，可以到opus官网：https://www.opus-codec.org 。



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

## 下载执行

在执行完上述命令后，会在build目录下生成可执行文件。
拷贝到视觉模块上运行：
```bash
chmod 777 ./Opus
./Opus <input.pcm> <output.pcm>
```