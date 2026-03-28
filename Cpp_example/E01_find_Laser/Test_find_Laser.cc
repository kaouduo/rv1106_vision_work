#include <lockzhiner_vision_module/edit/edit.h>
#include <lockzhiner_vision_module/vision/deep_learning/detection/paddle_det.h>
#include <lockzhiner_vision_module/vision/utils/visualize.h>

#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <queue>
#include <vector>

using namespace std::chrono;

// 绘制十字标记的函数
void drawCrossMarker(cv::Mat& image, cv::Point center,
                     cv::Scalar color = cv::Scalar(255, 0, 0), int size = 10,
                     int thickness = 2) {
  // 绘制水平线
  cv::line(image, cv::Point(center.x - size, center.y),
           cv::Point(center.x + size, center.y), color, thickness);

  // 绘制垂直线
  cv::line(image, cv::Point(center.x, center.y - size),
           cv::Point(center.x, center.y + size), color, thickness);
}

// 检测画面中最大的外矩形框
bool detectOuterRect(cv::Mat& frame, std::vector<cv::Point>& outerRect,
                     double& area) {
  // 转换为灰度图像
  cv::Mat gray;
  cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

  // 应用高斯模糊减少噪声
  cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);

  // Canny边缘检测
  cv::Mat edges;
  cv::Canny(gray, edges, 50, 150, 3);

  // 查找轮廓
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(edges, contours, hierarchy,
                   cv::RETR_EXTERNAL,  // 只检测外轮廓
                   cv::CHAIN_APPROX_SIMPLE);

  double maxArea = 0;
  std::vector<cv::Point> largestRect;

  // 查找最大的矩形轮廓（外矩形）
  for (size_t i = 0; i < contours.size(); i++) {
    // 近似多边形
    std::vector<cv::Point> approx;
    double epsilon = 0.02 * cv::arcLength(contours[i], true);
    cv::approxPolyDP(contours[i], approx, epsilon, true);

    // 筛选四边形
    if (approx.size() == 4 && cv::isContourConvex(approx)) {
      double currentArea = cv::contourArea(approx);
      if (currentArea > maxArea) {
        maxArea = currentArea;
        largestRect = approx;
      }
    }
  }

  // 返回检测结果
  if (maxArea > 1000) {
    outerRect = largestRect;
    area = maxArea;
    return true;
  }
  return false;
}

// 计算矩形的中心点
cv::Point calculateRectCenter(const std::vector<cv::Point>& rect) {
  cv::Point center(0, 0);
  for (const auto& pt : rect) {
    center.x += pt.x;
    center.y += pt.y;
  }
  center.x /= 4;
  center.y /= 4;
  return center;
}

// 检查两个矩形是否相似（中心点位置和面积）
bool areRectsSimilar(const std::vector<cv::Point>& rect1,
                     const std::vector<cv::Point>& rect2, double area1,
                     double area2) {
  // 计算中心点
  cv::Point center1 = calculateRectCenter(rect1);
  cv::Point center2 = calculateRectCenter(rect2);

  // 计算中心点距离
  double distance = cv::norm(center1 - center2);

  // 计算面积差异
  double areaDiff = std::fabs(area1 - area2);
  double areaRatio = areaDiff / std::max(area1, area2);

  // 判断是否相似（阈值可根据实际调整）
  return (distance < 20) && (areaRatio < 0.1);
}

// 绘制固定的外矩形标记
void drawFixedRectMarkers(cv::Mat& frame,
                          const std::vector<cv::Point>& outerRect,
                          double area) {
  // 绘制外矩形框
  cv::polylines(frame, std::vector<std::vector<cv::Point>>{outerRect}, true,
                cv::Scalar(255, 0, 0), 2);
}

int main(int argc, char* argv[]) {
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
  cap.open(0);

  if (!cap.isOpened()) {
    std::cerr << "Error: Could not open camera." << std::endl;
    return 1;
  }

  // 状态变量
  bool rectDetected = false;
  std::vector<cv::Point> calibratedOuterRect;
  double calibratedOuterArea = 0;
  int similarDetectionCount = 0;
  const int requiredSimilarDetections = 5;
  auto startTime = high_resolution_clock::now();

  // 存储最近检测到的矩形（用于比较）
  struct RectDetection {
    std::vector<cv::Point> points;
    double area;
  };
  RectDetection lastDetection;

  cv::Mat input_mat;
  while (true) {
    // 捕获一帧图像
    cap >> input_mat;
    if (input_mat.empty()) {
      std::cerr << "Warning: Captured an empty frame." << std::endl;
      continue;
    }

    // 创建原始图像的副本用于模型推理
    cv::Mat inference_mat = input_mat.clone();

    auto currentTime = high_resolution_clock::now();
    auto elapsedTime = duration_cast<seconds>(currentTime - startTime).count();

    // 5秒后开始尝试检测矩形
    if (elapsedTime > 5 && !rectDetected) {
      // 检测外矩形框
      std::vector<cv::Point> outerRect;
      double outerArea = 0;

      bool detected = detectOuterRect(input_mat, outerRect, outerArea);

      if (detected) {
        // 如果是第一次检测到，保存为参考
        if (similarDetectionCount == 0) {
          lastDetection.points = outerRect;
          lastDetection.area = outerArea;
          similarDetectionCount = 1;
          std::cout << "Initial rectangle detected. Starting verification..."
                    << std::endl;
        }
        // 与上一次检测比较
        else if (areRectsSimilar(lastDetection.points, outerRect,
                                 lastDetection.area, outerArea)) {
          similarDetectionCount++;
          std::cout << "Matching rectangle detected (" << similarDetectionCount
                    << "/" << requiredSimilarDetections << ")" << std::endl;
        }
        // 不相似，重置计数器
        else {
          similarDetectionCount = 1;
          lastDetection.points = outerRect;
          lastDetection.area = outerArea;
          std::cout << "Rectangle changed, resetting count." << std::endl;
        }

        // 如果连续检测到相同矩形达到5次，确认标定
        if (similarDetectionCount >= requiredSimilarDetections) {
          rectDetected = true;
          calibratedOuterRect = outerRect;
          calibratedOuterArea = outerArea;
          std::cout << "Rectangle calibration complete! Using last detected "
                       "rectangle."
                    << std::endl;
        }
      }
    }

    // 如果已检测到外矩形，则长期标注它
    if (rectDetected) {
      drawFixedRectMarkers(input_mat, calibratedOuterRect, calibratedOuterArea);
    }

    // 使用复制的图像进行模型推理（确保没有绘制标记）
    auto start_time = high_resolution_clock::now();
    auto results = model.Predict(inference_mat);
    auto end_time = high_resolution_clock::now();

    // 计算推理时间
    auto time_span = duration_cast<milliseconds>(end_time - start_time);
    std::cout << "Inference time: " << time_span.count() << " ms" << std::endl;

    // 显示检测数量信息
    int totalDetections = results.size();

    // 绘制所有检测结果（移除置信度过滤）
    for (const auto& result : results) {
      // 绘制目标框（蓝色矩形）
      // cv::rectangle(input_mat,
      //               cv::Rect(result.box.x, result.box.y, result.box.width,
      //                        result.box.height),
      //               cv::Scalar(255, 0, 0),  // BGR颜色：蓝色
      //               2);                     // 线宽

      // 计算目标框中心点
      cv::Point box_center(result.box.x + result.box.width / 2,
                           result.box.y + result.box.height / 2);

      // 绘制目标框中心十字标记（蓝色）
      drawCrossMarker(input_mat, box_center, cv::Scalar(255, 0, 0), 10, 2);

      // 创建置信度文本（保留两位小数）
      std::string confText =
          "Conf: " + std::to_string(static_cast<int>(result.score * 100)) + "%";

      // 显示坐标信息
      std::string coordText = "(" + std::to_string(box_center.x) + ", " +
                              std::to_string(box_center.y) + ")";
      cv::putText(input_mat, coordText,
                  cv::Point(box_center.x + 10, box_center.y - 10),
                  cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 1);

      // 在目标框左上角显示置信度
      cv::putText(input_mat, confText,
                  cv::Point(result.box.x, result.box.y - 5),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);
    }

    // 在控制台输出检测统计信息
    if (totalDetections > 0) {
      std::cout << "Detected " << totalDetections << " objects" << std::endl;
    }

    // 打印带标注的图像
    edit.Print(input_mat);
  }

  cap.release();
  return 0;
}