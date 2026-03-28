#include <lockzhiner_vision_module/edit/edit.h>
#include <lockzhiner_vision_module/vision/deep_learning/detection/paddle_det.h>

#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std::chrono;

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: Test-PaddleDet model_path" << std::endl;
    return 1;
  }

  // 初始化模型
  lockzhiner_vision_module::vision::PaddleDet model;
  if (!model.Initialize(argv[1])) {
    std::cout << "Failed to initialize model." << std::endl;
    return 1;
  }

  lockzhiner_vision_module::edit::Edit edit;
  if (!edit.StartAndAcceptConnection()) {
    std::cerr << "Error: Failed to start and accept connection." << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << "Device connected successfully." << std::endl;

  // 打开摄像头
  cv::VideoCapture cap;
  cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

  if (!cap.open(0)) {
    std::cerr << "Couldn't open video capture device" << std::endl;
    return -1;
  }

  cv::Mat input_mat;

  // 定义标签映射表
  const std::vector<std::string> label_map = {"5", "8", "4", "3",
                                              "7", "6", "2", "1"};

  while (true) {
    cap >> input_mat;
    if (input_mat.empty()) {
      std::cerr << "Warning: Captured an empty frame." << std::endl;
      continue;
    }

    // 推理
    high_resolution_clock::time_point start_time = high_resolution_clock::now();
    auto results = model.Predict(input_mat);
    high_resolution_clock::time_point end_time = high_resolution_clock::now();

    auto time_span = duration_cast<milliseconds>(end_time - start_time);
    std::cout << "Inference time: " << time_span.count() << " ms" << std::endl;

    // 手动绘制检测结果
    cv::Mat output_image = input_mat.clone();  // 复制原始图像用于绘制

    for (const auto &result : results) {
      int label_id = result.label_id;
      float score = result.score;
      cv::Rect bbox = result.box;

      // 映射 label_id 到实际标签
      std::string label =
          (label_id >= 0 && label_id < static_cast<int>(label_map.size()))
              ? label_map[label_id]
              : "unknown";

      // 绘制矩形框
      cv::rectangle(output_image, bbox, cv::Scalar(0, 255, 0), 2);  // 绿色框

      // 构造显示文本
      std::string text = label + " " + cv::format("%.2f", score);

      // 设置字体
      int font_face = cv::FONT_HERSHEY_SIMPLEX;
      double font_scale = 0.5;
      int thickness = 1;

      // 获取文本大小
      int baseline;
      cv::Size text_size =
          cv::getTextSize(text, font_face, font_scale, thickness, &baseline);

      // 计算文本背景区域
      cv::Rect text_rect(bbox.x, bbox.y - text_size.height, text_size.width,
                         text_size.height + baseline);
      text_rect &=
          cv::Rect(0, 0, output_image.cols, output_image.rows);  // 避免越界

      // 填充文本背景
      cv::rectangle(output_image, text_rect, cv::Scalar(0, 255, 0), cv::FILLED);

      // 绘制文本
      cv::putText(output_image, text, cv::Point(bbox.x, bbox.y), font_face,
                  font_scale,
                  cv::Scalar::all(0),  // 黑色文字
                  thickness, 8);
    }

    // 发送到设备显示
    edit.Print(output_image);
  }

  cap.release();
  return 0;
}