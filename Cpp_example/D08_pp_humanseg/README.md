# 图像分割
本章节在 Lockzhiner Vision Module 上基于paddleseg模型, 实现了一个PP-HumanSeg人像分割案例。
## 1. 基本知识简介
### 1.1 人像分割简介
人像分割是一种基于计算机视觉的技术，通过深度学习算法精准识别图像或视频中的人物主体，将其与背景进行像素级分离。该技术可实时运行于移动端及嵌入式设备，广泛应用于虚拟背景、智能抠图、视频会议美颜等场景，支持复杂光照、多样姿态和遮挡情况下的高精度分割，兼顾处理速度与效果。
### 1.2 人像分割常用方法
目前对于实现人像分割任务的方法有很多，下面介绍几种常用的人像分割实现方法。
- ​​传统算法（如GrabCut）​：基于颜色直方图与图割优化，适合简单背景，计算量小但精度有限。
- U-Net系列：编码器-解码器结构，医学图像起家，适合精细边缘，需较高算力。
- ​DeepLab系列：采用空洞卷积扩大感受野，擅长复杂场景，模型较大。
- ​​BiSeNet​：双分支结构平衡速度与精度，实时分割首选，移动端友好。
- ​​PP-HumanSeg​：百度自研轻量模型，专为人像优化，支持半监督训练，RKNN部署效率高。

这些方法各有优势，其中在工业部署方面PP-HumanSeg（精度与速度平衡）和BiSeNet（高性价比）更适合，可配合OpenCV后处理优化边缘。

---

## 2. C++ API 文档
### 2.1 RKNPU2Backend 类
#### 2.1.1 头文件
```cpp
#include "rknpu2_backend/rknpu2_backend.h"
```
- 作用：创建一个RKNPU2Backend类，用于实现对rknn模型的处理。

#### 2.1.2 构造类函数
```cpp
ockzhiner_vision_module::vision::RKNPU2Backend backend;
```
- 作用：创建一个RKNPU2Backen类型的对象实例，用于实现人像分割。
- 参数说明：
    - 无
- 返回值：
    - 无

#### 2.1.3 Initialize 函数
```cpp
bool Initialize(const std::string &model_path, const std::string &param_path = "") override;
```
- 作用：初始化 RKNN 模型，加载模型文件和可选参数文件，完成推理引擎的准备工作。
- 参数说明：
    - model_path：必需参数，RKNN 模型文件路径（.rknn 格式）。
    - param_path：可选参数，额外参数文件路径（某些场景下用于补充模型配置，默认空字符串）。
- 返回值：返回ture/false，表示模型初始化是否成功。

#### 2.1.4 Run 函数
```cpp
bool Run();
```
- 作用：执行模型推理计算，驱动输入数据通过模型计算得到输出结果。
- 参数说明：
    - 无
- 返回值：
    - true：推理执行成功。
    - false：推理失败（可能原因：输入数据未准备、内存不足等）。

## 3. PP-Humanseg人像分割代码解析
### 3.1 流程图

<img src="./images/view.png" width="500" height="1200">

### 3.2 核心代码解析
- 初始化模型
```cpp
backend.Initialize(model_path)
```
- 获取输入输出属性
```cpp
const auto& input_tensor = backend.GetInputTensor(0);
const auto& output_tensor = backend.GetOutputTensor(0);
```
- 对输入图像进行推理
```cpp
backend.Run()
```
自定义函数说明
- pp-humanseg输入预处理
```cpp
cv::Mat preprocess(const cv::Mat& image, 
                const std::vector<size_t>& input_dims,
                float input_scale,
                int input_zp)
```
- 作用：对输入图像进行预处理操作，包括​尺寸调整​​、​​颜色空间转换​​和​​量化处理​​，使其符合RKNN模型的输入要求。
- 参数说明：
    - image：输入图像。
    - input_dims：模型输入张量的维度定义（需满足 [1, H, W, 3] 的 NHWC 格式）。
    - input_scale：量化缩放因子（用于将浮点像素值转换为 INT8 数值）。
    - input_zp：量化零点偏移值（INT8 数值的偏移基准）。
- 返回值：
    - 返回 HxWx3 的量化张量（cv::Mat，数据类型为 CV_8S）。
    - 若输入维度不合法，返回空矩阵（cv::Mat()）并报错。

- pp-humanseg输入后处理
```cpp
cv::Mat postprocess(const int8_t* output_data, 
                   const std::vector<size_t>& output_dims,
                   float output_scale,
                   int output_zp,
                   const cv::Size& target_size)
```
- 作用：将RKNN模型的​​量化输出​​转换为​高精度分割掩模​​，通过多阶段优化（概率生成、阈值分割、形态学处理、边缘优化）生成与输入图像尺寸匹配的二进制掩模。
- 参数说明：
    - output_data：模型输出的量化数据指针。
    - output_dims：输出张量维度，需满足[1, 2, H, W]的NCHW格式。
    - output_scale：反量化缩放因子。
    - output_zp：反量化零点偏移值。
    - taeget_size：目标输出尺寸。
- 返回值：处理后的二进制掩模。

### 3.3 完整代码实现
```cpp
#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "rknpu2_backend/rknpu2_backend.h"
#include <chrono>
#include <cstdlib> 
#include <ctime> 

using namespace std::chrono;

// 预处理函数
cv::Mat preprocess(const cv::Mat& image, 
                  const std::vector<size_t>& input_dims,
                  float input_scale,
                  int input_zp) {
    // 确保输入维度为NHWC [1, H, W, 3]
    if (input_dims.size() != 4 || input_dims[0] != 1 || input_dims[3] != 3) {
        std::cerr << "Invalid input dimensions" << std::endl;
        return cv::Mat();
    }

    const size_t input_h = input_dims[1];
    const size_t input_w = input_dims[2];

    // Resize并转换颜色空间
    cv::Mat resized, rgb;
    cv::resize(image, resized, cv::Size(input_w, input_h));
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    // 量化到INT8
    cv::Mat quantized;
    float scale = 1.0f / (input_scale * 255.0f);
    rgb.convertTo(quantized, CV_8S, scale, input_zp);

    return quantized;
}

// 后处理函数
cv::Mat postprocess(const int8_t* output_data, 
                   const std::vector<size_t>& output_dims,
                   float output_scale,
                   int output_zp,
                   const cv::Size& target_size) {
    // 验证输出维度为NCHW [1, 2, H, W]
    if (output_dims.size() != 4 || output_dims[0] != 1 || output_dims[1] != 2) {
        std::cerr << "Invalid output dimensions" << std::endl;
        return cv::Mat();
    }

    const int h = output_dims[2];
    const int w = output_dims[3];

    // ================= 1. 概率图生成优化 =================
    cv::Mat prob_map(h, w, CV_32FC1);
    float spatial_weight = 1.0f - (h * w) / (192.0f * 192.0f); 
    
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int bg_idx = 0 * h * w + y * w + x;
            const int fg_idx = 1 * h * w + y * w + x;
            
            float bg_logit = std::clamp((output_data[bg_idx] - output_zp) * output_scale, -10.0f, 10.0f);
            float fg_logit = std::clamp((output_data[fg_idx] - output_zp) * output_scale, -10.0f, 10.0f);

            float center_weight = 1.0f - (std::abs(x - w/2.0f)/(w/2.0f) + std::abs(y - h/2.0f)/(h/2.0f))/2.0f;
            fg_logit *= (1.2f + 0.3f * center_weight * spatial_weight);

            float max_logit = std::max(bg_logit, fg_logit);
            float exp_sum = expf(bg_logit - max_logit) + expf(fg_logit - max_logit);
            prob_map.at<float>(y, x) = expf(fg_logit - max_logit) / (exp_sum + 1e-8f);
        }
    }

    // ================= 2. 自适应阈值优化 =================
    cv::Mat binary_mask;
    cv::Mat prob_roi = prob_map(cv::Rect(w/4, h/4, w/2, h/2)); 
    float center_mean = cv::mean(prob_roi)[0];
    float dynamic_thresh = std::clamp(0.45f - (center_mean - 0.5f) * 0.3f, 0.25f, 0.65f);
    
    cv::threshold(prob_map, binary_mask, dynamic_thresh, 255, cv::THRESH_BINARY);
    binary_mask.convertTo(binary_mask, CV_8U);

    // ================= 3. 多尺度形态学处理 =================
    std::vector<cv::Mat> mask_pyramid;
    cv::buildPyramid(binary_mask, mask_pyramid, 2); 
    
    cv::Mat kernel1 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3));
    cv::morphologyEx(mask_pyramid[1], mask_pyramid[1], cv::MORPH_OPEN, kernel1);
    
    cv::Mat kernel2 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5));
    cv::morphologyEx(mask_pyramid[0], mask_pyramid[0], cv::MORPH_CLOSE, kernel2);
    
    cv::Mat refined_mask;
    cv::pyrUp(mask_pyramid[1], refined_mask, mask_pyramid[0].size());
    cv::bitwise_and(refined_mask, mask_pyramid[0], refined_mask);

    // ================= 4. 智能边缘优化 =================
    cv::Mat edge_weights;
    cv::distanceTransform(refined_mask, edge_weights, cv::DIST_L2, 3);
    cv::normalize(edge_weights, edge_weights, 0, 1.0, cv::NORM_MINMAX);
    
    cv::Mat probabilistic_edges;
    cv::Canny(refined_mask, probabilistic_edges, 50, 150);
    probabilistic_edges.convertTo(probabilistic_edges, CV_32F, 1.0/255.0);
    
    cv::Mat final_edges;
    cv::multiply(probabilistic_edges, edge_weights, final_edges);
    final_edges.convertTo(final_edges, CV_8U, 255.0);

    // ================= 5. 多模态融合输出 =================
    cv::Mat resized_mask;
    cv::resize(refined_mask, resized_mask, target_size, 0, 0, cv::INTER_LANCZOS4);
    
    cv::Mat final_mask;
    cv::bilateralFilter(resized_mask, final_mask, 5, 15, 15);
    
    cv::Mat edge_mask_hr;
    cv::resize(final_edges, edge_mask_hr, target_size, 0, 0, cv::INTER_NEAREST);
    final_mask.setTo(255, edge_mask_hr > 128);

    return final_mask;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <image_path>" << std::endl;
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string image_path = argv[2];

    // 初始化RKNN后端
    lockzhiner_vision_module::vision::RKNPU2Backend backend;
    if (!backend.Initialize(model_path)) {
        std::cerr << "Failed to initialize RKNN backend" << std::endl;
        return -1;
    }

    // 加载图像
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Failed to read image: " << image_path << std::endl;
        return -1;
    }

    // 获取输入Tensor
    const auto& input_tensor = backend.GetInputTensor(0);
    std::vector<size_t> input_dims = input_tensor.GetDims();
    float input_scale = input_tensor.GetScale();
    int input_zp = input_tensor.GetZp();

    // 预处理
    cv::Mat preprocessed = preprocess(image, input_dims, input_scale, input_zp);
    if (preprocessed.empty()) {
        std::cerr << "Preprocessing failed" << std::endl;
        return -1;
    }

    // 验证输入数据尺寸
    size_t expected_input_size = input_tensor.GetElemsBytes();
    size_t actual_input_size = preprocessed.total() * preprocessed.elemSize();
    if (expected_input_size != actual_input_size) {
        std::cerr << "Input size mismatch! Expected: " << expected_input_size
                  << ", Actual: " << actual_input_size << std::endl;
        return -1;
    }

    // 拷贝输入数据
    void* input_data = input_tensor.GetData();
    memcpy(input_data, preprocessed.data, actual_input_size);

    // 执行推理
    high_resolution_clock::time_point start_time = high_resolution_clock::now();
    if (!backend.Run()) {
        std::cerr << "Inference failed" << std::endl;
        return -1;
    }
    
    // 获取输出Tensor
    const auto& output_tensor = backend.GetOutputTensor(0);
    std::vector<size_t> output_dims = output_tensor.GetDims();
    float output_scale = output_tensor.GetScale();
    int output_zp = output_tensor.GetZp();
    const int8_t* output_data = static_cast<const int8_t*>(output_tensor.GetData());

    // 后处理
    high_resolution_clock::time_point end_time = high_resolution_clock::now();
    auto time_span = duration_cast<milliseconds>(end_time - start_time);
    cv::Mat mask = postprocess(output_data, output_dims, output_scale, output_zp, image.size());
    
    std::cout << "单张图片推理时间(ms): " << time_span.count() << std::endl;

    // 生成结果
    cv::Mat result;
    cv::bitwise_and(image, image, result, mask);

    // 保存结果
    const std::string output_path = "result.jpg";
    cv::imwrite(output_path, result);
    std::cout << "Result saved to: " << output_path << std::endl;

    // 显示调试视图
    cv::imshow("Original", image);
    cv::imshow("Mask", mask);
    cv::imshow("Result", result);
    cv::waitKey(0);

    return 0;
}
```

---

## 4. 编译调试
### 4.1 编译环境搭建
- 请确保你已经按照 [开发环境搭建指南](../../../../docs/introductory_tutorial/cpp_development_environment.md) 正确配置了开发环境。
- 同时已经正确连接开发板。
### 4.2 Cmake介绍
```cmake
cmake_minimum_required(VERSION 3.10)

project(pp_humanseg)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 定义项目根目录路径
set(PROJECT_ROOT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../..")
message("PROJECT_ROOT_PATH = " ${PROJECT_ROOT_PATH})

include("${PROJECT_ROOT_PATH}/toolchains/arm-rockchip830-linux-uclibcgnueabihf.toolchain.cmake")

# 定义 OpenCV SDK 路径
set(OpenCV_ROOT_PATH "${PROJECT_ROOT_PATH}/third_party/opencv-mobile-4.10.0-lockzhiner-vision-module")
set(OpenCV_DIR "${OpenCV_ROOT_PATH}/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)
set(OPENCV_LIBRARIES "${OpenCV_LIBS}")

# 定义 LockzhinerVisionModule SDK 路径
set(LockzhinerVisionModule_ROOT_PATH "${PROJECT_ROOT_PATH}/third_party/lockzhiner_vision_module_sdk")
set(LockzhinerVisionModule_DIR "${LockzhinerVisionModule_ROOT_PATH}/lib/cmake/lockzhiner_vision_module")
find_package(LockzhinerVisionModule REQUIRED)

# 配置rknpu2
set(RKNPU2_BACKEND_BASE_DIR "${LockzhinerVisionModule_ROOT_PATH}/include/lockzhiner_vision_module/vision/deep_learning/runtime")
if(NOT EXISTS ${RKNPU2_BACKEND_BASE_DIR})
    message(FATAL_ERROR "RKNPU2 backend base dir missing: ${RKNPU2_BACKEND_BASE_DIR}")
endif()


add_executable(Test-pp_humanseg pp_humanseg.cc)
target_include_directories(Test-pp_humanseg PRIVATE ${LOCKZHINER_VISION_MODULE_INCLUDE_DIRS}  ${rknpu2_INCLUDE_DIRS} ${RKNPU2_BACKEND_BASE_DIR})
target_link_libraries(Test-pp_humanseg PRIVATE ${OPENCV_LIBRARIES} ${NCNN_LIBRARIES} ${LOCKZHINER_VISION_MODULE_LIBRARIES})

install(
    TARGETS Test-pp_humanseg
    RUNTIME DESTINATION .  
)
```
### 4.3 编译项目
使用 Docker Destop 打开 LockzhinerVisionModule 容器并执行以下命令来编译项目。
```bash
# 进入Demo所在目录
cd /LockzhinerVisionModuleWorkSpace/LockzhinerVisionModule/Cpp_example/D08_pp_humanseg
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

---

## 5. 执行结果
### 5.1 运行前准备
- 请确保你已经下载了 [凌智视觉模块人像分割模型](https://gitee.com/LockzhinerAI/LockzhinerVisionModule/releases/download/v0.0.6/pp-humanseg.rknn)
### 5.2 运行过程
```shell
chmod 777 Test-pp_humanseg
# 对人像进行分割
./Test-pp_humanseg pp-humanseg.rknn image_path
```
### 5.3 运行效果
#### 5.3.1 人像分割结果
- 原始图像

![title](./images/test.png)

- 分割结果

![title](./images/result.jpg)

#### 5.3.2 注意事项
由于分割的模型很小，并且在模型转换过程中会有精度损失，所以在测试的时候尽量选择背景比较纯净的图像效果比较好。

---

## 6. 总结
通过上述内容，我们成功实现了一个简单的人像分割的例子，包括：

- 加载图像分割的rknn模型和待分割图像。
- 图像预处理和模型推理。
- 图像后处理并保存分割结果。