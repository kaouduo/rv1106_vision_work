#include <lockzhiner_vision_module/edit/edit.h>

#include <Eigen/Dense>
#include <iostream>
#include <opencv2/core/eigen.hpp>

#include "opencv2/opencv.hpp"

// === 可选标签家族头文件（取消注释对应家族以切换） ===
extern "C" {
#include "apriltag.h"
#include "apriltag_pose.h"

// 可选标签家族（根据需求取消注释）：
// 1. tag36h11（默认） - 高鲁棒性 36x36 黑白图案
#include "tag36h11.h"

// 2. tag25h9 - 25x25 黑白图案
// #include "tag25h9.h"

// 3. tag16h5 - 16x16 黑白图案
// #include "tag16h5.h"

// 4. tagCircle21h7 - 圆形标签
// #include "tagCircle21h7.h"

// 5. tagCircle49h12 - 大尺寸圆形标签
// #include "tagCircle49h12.h"

// 6. tagStandard41h12 - 标准尺寸标签
// #include "tagStandard41h12.h"

// 7. tagStandard52h13 - 大尺寸标准标签
// #include "tagStandard52h13.h"

// 8. tagCustom48h12 - 自定义标签
// #include "tagCustom48h12.h"
}

// 手动投影函数
void manualProjectPoint(const cv::Point3d &point, const cv::Mat &R,
                        const cv::Mat &t, cv::Point2d &projected, double fx,
                        double fy, double cx, double cy) {
  Eigen::Vector4d point_3d(point.x, point.y, point.z, 1);

  Eigen::Matrix<double, 3, 4> Rt;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      Rt(i, j) = R.at<double>(i, j);
    }
    Rt(i, 3) = t.at<double>(i, 0);
  }

  Eigen::Vector3d proj_point = Rt * point_3d;
  double x = proj_point(0) / proj_point(2);
  double y = proj_point(1) / proj_point(2);

  projected.x = fx * x + cx;
  projected.y = fy * y + cy;
}

// Function to draw a 3D cube around the detected tag
void drawCube(cv::Mat &frame, const apriltag_pose_t &pose,
              const apriltag_detection_info_t &info) {
  double size = info.tagsize / 2.0;
  std::vector<cv::Point3d> points = {
      {-size, -size, 0},         {size, -size, 0},
      {size, size, 0},           {-size, size, 0},
      {-size, -size, -2 * size}, {size, -size, -2 * size},
      {size, size, -2 * size},   {-size, size, -2 * size}};

  cv::Mat rvec(3, 3, CV_64FC1, pose.R->data);  // Rotation matrix
  cv::Mat tvec(3, 1, CV_64FC1, pose.t->data);  // Translation vector

  std::vector<cv::Point2d> imgPoints(points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    manualProjectPoint(points[i], rvec, tvec, imgPoints[i], info.fx, info.fy,
                       info.cx, info.cy);
  }

  // Draw lines of the cube
  for (int i = 0; i < 4; ++i) {
    cv::line(frame, imgPoints[i], imgPoints[(i + 1) % 4], cv::Scalar(0, 255, 0),
             1);  // Top face
    cv::line(frame, imgPoints[i + 4], imgPoints[(i + 1) % 4 + 4],
             cv::Scalar(0, 255, 0),
             1);  // Bottom face
    cv::line(frame, imgPoints[i], imgPoints[i + 4], cv::Scalar(0, 255, 0),
             1);  // Vertical lines
  }
}

int main() {
  // === 可配置参数（通过注释说明） ===

  // 标签家族选择（需配合 extern "C" 中的头文件使用）
  // 1. tag36h11 - 已包含
  // 2. tag25h9 - 取消注释 #include "tag25h9.h" 并修改 tf = tag25h9_create()
  // 3. tag16h5 - 取消注释 #include "tag16h5.h" 并修改 tf = tag16h5_create()
  // 其他同理...

  // 标签尺寸（单位：米）
  double tag_size = 0.146 - 0.012 * 4;  // 根据实际标签尺寸修改

  // 相机内参（需根据实际相机标定结果修改）
  double fx = 848.469;  // 焦距x
  double fy = 847.390;  // 焦距y
  double cx = 160;      // 光心x（图像宽度的一半）
  double cy = 120;      // 光心y（图像高度的一半）

  // 检测器配置参数
  int num_threads = 1;       // 使用的线程数
  double decimate = 2.0;     // 输入图像降采样因子
  double blur = 0.0;         // 模糊强度
  bool refine_edges = true;  // 是否优化边缘检测

  // 坐标系选择（true=世界坐标系，false=相机坐标系）
  bool use_world_coords = true;

  // === 初始化 ===
  cv::VideoCapture cap;
  cap.set(cv::CAP_PROP_FRAME_WIDTH, 320);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 240);
  if (!cap.open(0)) {
    std::cerr << "Couldn't open video capture device" << std::endl;
    return -1;
  }

  // 创建标签家族实例（根据选择切换函数）
  // 默认使用 tag36h11：
  apriltag_family_t *tf = tag36h11_create();
  // 切换到 tag25h9：
  // apriltag_family_t *tf = tag25h9_create();
  // 切换到 tag16h5：
  // apriltag_family_t *tf = tag16h5_create();
  // 其他同理...

  apriltag_detector_t *td = apriltag_detector_create();
  apriltag_detector_add_family(td, tf);
  td->quad_decimate = decimate;
  td->quad_sigma = blur;
  td->nthreads = num_threads;
  td->refine_edges = refine_edges;

  apriltag_detection_info_t info;
  info.tagsize = tag_size;
  info.fx = fx;
  info.fy = fy;
  info.cx = cx;
  info.cy = cy;

  lockzhiner_vision_module::edit::Edit edit;
  if (!edit.StartAndAcceptConnection()) {
    std::cerr << "Error: Failed to start and accept connection." << std::endl;
    return EXIT_FAILURE;
  }

  // === 主循环 ===
  cv::Mat frame, gray;
  while (true) {
    cap >> frame;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    image_u8_t im = {.width = gray.cols,
                     .height = gray.rows,
                     .stride = gray.cols,
                     .buf = gray.data};

    zarray_t *detections = apriltag_detector_detect(td, &im);

    for (int i = 0; i < zarray_size(detections); i++) {
      apriltag_detection_t *det;
      zarray_get(detections, i, &det);
      info.det = det;
      apriltag_pose_t pose;
      estimate_tag_pose(&info, &pose);

      if (use_world_coords) {
        // 世界坐标系下的位置
        cv::Mat rvec(3, 3, CV_64FC1, pose.R->data);
        cv::Mat tvec(3, 1, CV_64FC1, pose.t->data);
        cv::Mat Pos = rvec.inv() * tvec;
        std::cout << "Tx: " << Pos.ptr<double>(0)[0] << std::endl;
        std::cout << "Ty: " << Pos.ptr<double>(1)[0] << std::endl;
        std::cout << "Tz: " << Pos.ptr<double>(2)[0] << std::endl;

        // 计算欧拉角（ZYX 顺序）
        double R11 = rvec.at<double>(0, 0), R12 = rvec.at<double>(0, 1),
               R13 = rvec.at<double>(0, 2);
        double R21 = rvec.at<double>(1, 0), R22 = rvec.at<double>(1, 1),
               R23 = rvec.at<double>(1, 2);
        double R31 = rvec.at<double>(2, 0), R32 = rvec.at<double>(2, 1),
               R33 = rvec.at<double>(2, 2);

        double roll = std::atan2(R32, R33);
        double pitch = std::asin(-R31);
        double yaw = std::atan2(R21, R11);

        std::cout << "Rx: " << roll * 180 / CV_PI << "°" << std::endl;
        std::cout << "Ry: " << pitch * 180 / CV_PI << "°" << std::endl;
        std::cout << "Rz: " << yaw * 180 / CV_PI << "°" << std::endl;
        std::cout << "-----------world--------------" << std::endl;
      } else {
        // 相机坐标系下的位置
        std::cout << "Tx: " << pose.t->data[0] << std::endl;
        std::cout << "Ty: " << pose.t->data[1] << std::endl;
        std::cout << "Tz: " << pose.t->data[2] << std::endl;

        cv::Mat rvec(3, 3, CV_64FC1, pose.R->data);
        double R11 = rvec.at<double>(0, 0), R12 = rvec.at<double>(0, 1),
               R13 = rvec.at<double>(0, 2);
        double R21 = rvec.at<double>(1, 0), R22 = rvec.at<double>(1, 1),
               R23 = rvec.at<double>(1, 2);
        double R31 = rvec.at<double>(2, 0), R32 = rvec.at<double>(2, 1),
               R33 = rvec.at<double>(2, 2);

        double roll = std::atan2(R32, R33);
        double pitch = std::asin(-R31);
        double yaw = std::atan2(R21, R11);

        std::cout << "Rx: " << roll * 180 / CV_PI << "°" << std::endl;
        std::cout << "Ry: " << pitch * 180 / CV_PI << "°" << std::endl;
        std::cout << "Rz: " << yaw * 180 / CV_PI << "°" << std::endl;
        std::cout << "-----------camera-------------" << std::endl;
      }

      // Draw the cube around the tag
      drawCube(frame, pose, info);

      // Display tag ID
      std::stringstream ss;
      ss << det->id;
      std::string text = ss.str();
      int fontface = cv::FONT_HERSHEY_SCRIPT_SIMPLEX;
      double fontscale = 1.0;
      int baseline;
      cv::Size textsize =
          cv::getTextSize(text, fontface, fontscale, 2, &baseline);
      cv::putText(frame, text,
                  cv::Point(det->c[0] - textsize.width / 2,
                            det->c[1] + textsize.height / 2),
                  fontface, fontscale, cv::Scalar(0xff, 0x99, 0), 2);
    }
    apriltag_detections_destroy(detections);

    edit.Print(frame);
    if (cv::waitKey(30) >= 0) break;
  }

  // === 清理资源 ===
  // 销毁对应的标签家族实例（需与创建函数匹配）
  tag36h11_destroy(tf);  // 当前使用 tag36h11
  // 切换到 tag25h9 时：
  // tag25h9_destroy(tf);
  // 切换到 tag16h5 时：
  // tag16h5_destroy(tf);
  // 其他同理...

  apriltag_detector_destroy(td);
  return 0;
}