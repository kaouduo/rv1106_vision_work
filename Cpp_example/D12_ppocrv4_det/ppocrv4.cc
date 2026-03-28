// #include <iostream>
// #include <cmath>
// #include <opencv2/opencv.hpp>
// #include "rknpu2_backend/rknpu2_backend.h"
// #include <cstdlib>
// #include <ctime>
// #include <stdio.h>
// #include <stdlib.h>
// #include <string.h>
// #include <math.h>
// #include "postprocess.h"
// #include <lockzhiner_vision_module/edit/edit.h>
// #include <lockzhiner_vision_module/vision/utils/visualize.h>
// int main(int argc, char *argv[])
// {
//     if (argc != 2)
//     {
//         std::cerr << "Usage: " << argv[0] << " <model_path> <image_path>" << std::endl;
//         return 1;
//     }

//     const std::string model_path = argv[1];


//     // 初始化RKNN后端
//     lockzhiner_vision_module::vision::RKNPU2Backend backend;
//     if (!backend.Initialize(model_path))
//     {
//         std::cerr << "Failed to initialize RKNN backend" << std::endl;
//         return -1;
//     }
//     lockzhiner_vision_module::edit::Edit edit;

//     if (!edit.StartAndAcceptConnection())
//     {
//         std::cerr << "Error: Failed to start and accept connection." << std::endl;
//         return EXIT_FAILURE;
//     }
//     std::cout << "Device connected successfully." << std::endl;
//     // 打开摄像头
//     cv::VideoCapture cap;
//     cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
//     cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
//     cap.open(0);
//     if (!cap.isOpened())
//     {
//         std::cerr << "Error: Could not open camera." << std::endl;
//         return 1;
//     }
//     cv::Mat image;
//     while (true)
//     {
//         cap >> image;

//         // 获取输入Tensor的信息
//         auto input_tensor = backend.GetInputTensor(0);
//         std::vector<size_t> input_dims = input_tensor.GetDims();
//         float input_scale = input_tensor.GetScale();
//         int input_zp = input_tensor.GetZp();

//         // 预处理
//         cv::Mat preprocessed = preprocess(image, input_dims, input_scale, input_zp);
//         if (preprocessed.empty())
//         {
//             std::cerr << "Preprocessing failed" << std::endl;
//             return -1;
//         }

//         // 验证输入数据尺寸
//         size_t expected_input_size = input_tensor.GetElemsBytes();
//         size_t actual_input_size = preprocessed.total() * preprocessed.elemSize();
//         if (expected_input_size != actual_input_size)
//         {
//             std::cerr << "Input size mismatch! Expected: " << expected_input_size
//                       << ", Actual: " << actual_input_size << std::endl;
//             return -1;
//         }
//         // 拷贝输入数据
//         void *input_data = input_tensor.GetData();
//         memcpy(input_data, preprocessed.data, actual_input_size);

//         // 推理
//         if (!backend.Run())
//         {
//             std::cerr << "Inference failed!" << std::endl;
//             free(input_data);
//             return -1;
//         }

//         // 获取输出结果
//         const auto &output_tensor = backend.GetOutputTensor(0);
//         std::vector<size_t> output_dims = output_tensor.GetDims();
//         float output_zp = output_tensor.GetZp();
//         float output_scale = output_tensor.GetScale();
//         const int8_t *output_data = static_cast<const int8_t *>(output_tensor.GetData());

//         // 后处理
//         ppocr_det_result results = {0};
//         const int8_t *output_data_int8 = static_cast<const int8_t *>(output_tensor.GetData());
//         std::vector<float> output_data_float(output_tensor.GetElemsBytes() / sizeof(int8_t));
//         for (size_t i = 0; i < output_tensor.GetNumElems(); ++i)
//         {
//             output_data_float[i] = (output_data_int8[i] - output_zp) * output_scale;
//         }

//         // 获取原始图像的宽高
//         int original_width = image.cols;
//         int original_height = image.rows;
//         float scale_w = (float)original_width / 480;
//         float scale_h = (float)original_height / 480;

//         dbnet_postprocess(output_data_float.data(),
//                           output_dims[2], output_dims[3], // det_out_w, det_out_h
//                           0.5, 0.3, true, "slow", 2.0, "quad",
//                           scale_w, scale_h, &results);

//         // 使用封装的绘图函数
//         draw_boxes(&image, results);

//         edit.Print(image);
//     }

//     cap.release();
//     return 0;
// }
#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "rknpu2_backend/rknpu2_backend.h"
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "postprocess.h"
#include <lockzhiner_vision_module/edit/edit.h>
#include <lockzhiner_vision_module/vision/utils/visualize.h>

// 用于计时的头文件
#include <chrono>

using namespace std::chrono;

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
        return 1;
    }

    const std::string model_path = argv[1];

    // 初始化RKNN后端
    lockzhiner_vision_module::vision::RKNPU2Backend backend;
    if (!backend.Initialize(model_path))
    {
        std::cerr << "Failed to initialize RKNN backend" << std::endl;
        return -1;
    }

    lockzhiner_vision_module::edit::Edit edit;

    if (!edit.StartAndAcceptConnection())
    {
        std::cerr << "Error: Failed to start and accept connection." << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Device connected successfully." << std::endl;

    // 打开摄像头
    cv::VideoCapture cap;
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.open(0);
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open camera." << std::endl;
        return 1;
    }

    cv::Mat image;
    int frame_count = 0; // 帧计数器

    while (true)
    {
        cap >> image;
        if (image.empty())
            continue;

        frame_count++;

        // 每隔3帧处理一次（即每4帧处理1次）
        if (frame_count % 4 == 1)
        {
            // 获取输入Tensor的信息
            auto input_tensor = backend.GetInputTensor(0);
            std::vector<size_t> input_dims = input_tensor.GetDims();
            float input_scale = input_tensor.GetScale();
            int input_zp = input_tensor.GetZp();

            // 预处理
            cv::Mat preprocessed = preprocess(image, input_dims, input_scale, input_zp);
            if (preprocessed.empty())
            {
                std::cerr << "Preprocessing failed" << std::endl;
                goto skip_inference;
            }

            // 验证输入数据尺寸
            size_t expected_input_size = input_tensor.GetElemsBytes();
            size_t actual_input_size = preprocessed.total() * preprocessed.elemSize();
            if (expected_input_size != actual_input_size)
            {
                std::cerr << "Input size mismatch! Expected: " << expected_input_size
                          << ", Actual: " << actual_input_size << std::endl;
                goto skip_inference;
            }

            // 拷贝输入数据
            void *input_data = input_tensor.GetData();
            memcpy(input_data, preprocessed.data, actual_input_size);

            // 开始计时
            auto start = high_resolution_clock::now();

            // 推理
            if (!backend.Run())
            {
                std::cerr << "Inference failed!" << std::endl;
                free(input_data);
                goto skip_inference;
            }

            // 结束计时
            auto end = high_resolution_clock::now();
            auto duration_ms = duration_cast<milliseconds>(end - start).count();
            std::cout << "Inference time: " << duration_ms << " ms" << std::endl;

            // 获取输出结果
            const auto &output_tensor = backend.GetOutputTensor(0);
            std::vector<size_t> output_dims = output_tensor.GetDims();
            float output_zp = output_tensor.GetZp();
            float output_scale = output_tensor.GetScale();
            const int8_t *output_data_int8 = static_cast<const int8_t *>(output_tensor.GetData());

            // 转换为浮点型
            std::vector<float> output_data_float(output_tensor.GetNumElems());
            for (size_t i = 0; i < output_tensor.GetNumElems(); ++i)
            {
                output_data_float[i] = (output_data_int8[i] - output_zp) * output_scale;
            }

            // 获取原始图像宽高
            int original_width = image.cols;
            int original_height = image.rows;
            float scale_w = (float)original_width / 480;
            float scale_h = (float)original_height / 480;

            // 后处理
            ppocr_det_result results = {0};
            dbnet_postprocess(output_data_float.data(),
                              output_dims[2], output_dims[3],
                              0.5, 0.3, true, "slow", 2.0, "quad",
                              scale_w, scale_h, &results);

            // 绘制检测框
            draw_boxes(&image, results);
        }

    skip_inference:
        // 显示当前帧（无论是否进行了推理）
        edit.Print(image);

        // 按下 ESC 键退出
        if (cv::waitKey(1) == 27)
        {
            break;
        }
    }

    cap.release();
    return 0;
}