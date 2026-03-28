#include <lockzhiner_vision_module/periphery/lcd/lcd_driver.h>

#include <iostream>
#include <opencv2/opencv.hpp>
int main() {
  // 设置视频采集参数（640x480分辨率）
  cv::VideoCapture cap;
  cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

  // 尝试打开摄像头设备（0号摄像头）
  if (!cap.open(0)) {
    std::cerr << "Camera failed to open\n";
    return 1;
  }
  if (lcd_init() < 0) {
    std::cerr << "LCD initialization failed" << std::endl;
    return -1;
  }
  cv::Mat frame;
  while (true) {
    cap >> frame;  // 从摄像头捕获一帧图像
    if (frame.empty()) continue;
    lcd_display_opencv_image(frame);
  }
}
