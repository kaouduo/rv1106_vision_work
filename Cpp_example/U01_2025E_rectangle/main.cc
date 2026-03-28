/**
 * U01_2025E_rectangle
 *
 * 功能：基于目标检测（PaddleDet）识别靶纸，在检测框内提取四边形轮廓，
 *       计算四个顶点的几何中心，并通过串口下发给单片机。
 *
 * 用法：
 *   ./Test-2025E-Rectangle <model_path> [--display]
 *
 *   --display  可选参数，启用上位机图像显示
 */

#include <lockzhiner_vision_module/vision/deep_learning/detection/paddle_det.h>
#include <lockzhiner_vision_module/edit/edit.h>
#include <lockzhiner_vision_module/periphery/usart/usart.h>
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>

using namespace std::chrono;

/**
 * @brief 简单的卡尔曼滤波器，用于追踪目标中心 (x, y) 及其速度 (vx, vy)
 */
class TargetTracker
{
public:
    TargetTracker() : kf(4, 2, 0), initialized(false), frames_lost(0)
    {
        // 状态转移矩阵 (A)
        // [ 1 0 dt 0 ]  x = x + vx * dt
        // [ 0 1 0 dt ]  y = y + vy * dt
        // [ 0 0 1  0 ]  vx = vx
        // [ 0 0 0  1 ]  vy = vy
        setIdentity(kf.transitionMatrix);

        // 测量矩阵 (H)
        // [ 1 0 0 0 ]  测量值为 x
        // [ 0 1 0 0 ]  测量值为 y
        kf.measurementMatrix = cv::Mat::zeros(2, 4, CV_32F);
        kf.measurementMatrix.at<float>(0, 0) = 1.0f;
        kf.measurementMatrix.at<float>(1, 1) = 1.0f;

        // 过程噪声协方差矩阵 (Q) - 适当减小，增强平滑度
        setIdentity(kf.processNoiseCov, cv::Scalar::all(5e-2));

        // 测量噪声协方差矩阵 (R) - 适当增大，减少抖动
        setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-2));

        // 后验错误协方差矩阵 (P)
        setIdentity(kf.errorCovPost, cv::Scalar::all(1.0));
    }

    void Initialize(cv::Point2f pt)
    {
        kf.statePost.at<float>(0) = pt.x;
        kf.statePost.at<float>(1) = pt.y;
        kf.statePost.at<float>(2) = 0;
        kf.statePost.at<float>(3) = 0;
        initialized = true;
        frames_lost = 0;
    }

    cv::Point2f Predict(float dt)
    {
        if (!initialized) return cv::Point2f(-1, -1);

        kf.transitionMatrix.at<float>(0, 2) = dt;
        kf.transitionMatrix.at<float>(1, 3) = dt;

        cv::Mat prediction = kf.predict();
        return cv::Point2f(prediction.at<float>(0), prediction.at<float>(1));
    }

    void Update(cv::Point2f pt)
    {
        if (!initialized)
        {
            Initialize(pt);
            return;
        }

        cv::Mat measurement = (cv::Mat_<float>(2, 1) << pt.x, pt.y);
        kf.correct(measurement);
        frames_lost = 0;
    }

    void MarkLost()
    {
        frames_lost++;
        if (frames_lost > 10) // 丢失超过 10 帧重置
        {
            initialized = false;
        }
    }

    bool IsInitialized() const { return initialized; }

private:
    cv::KalmanFilter kf;
    bool initialized;
    int frames_lost;
};

/**
 * @brief 计算两条直线 (p1-p2) 和 (p3-p4) 的交点
 * 在透视投影中，矩形的真实几何中心即为其对角线的交点
 */
cv::Point2f GetDiagonalIntersection(cv::Point2f p1, cv::Point2f p2, cv::Point2f p3, cv::Point2f p4)
{
    float x1 = p1.x, y1 = p1.y;
    float x2 = p2.x, y2 = p2.y;
    float x3 = p3.x, y3 = p3.y;
    float x4 = p4.x, y4 = p4.y;

    float denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
    if (std::abs(denominator) < 1e-6)
    {
        // 平行线，回退到四点平均值
        return cv::Point2f((x1 + x2 + x3 + x4) / 4.0f, (y1 + y2 + y3 + y4) / 4.0f);
    }

    float x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator;
    float y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator;

    return cv::Point2f(x, y);
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: Test-2025E-Rectangle <model_path> [--display]" << std::endl;
        return 1;
    }

    // ── 串口初始化 ──────────────────────────────────────────────────────────
    lockzhiner_vision_module::periphery::USART1 usart;
    if (!usart.Open(115200))
    {
        std::cerr << "Error: Failed to open USART." << std::endl;
        return 1;
    }

    // ── 参数解析（--display 开启上位机显示）──────────────────────────────────
    bool use_display = false;
    for (int i = 2; i < argc; i++)
    {
        if (std::string(argv[i]) == "--display")
        {
            use_display = true;
        }
    }

    // ── Edit 初始化（可选）───────────────────────────────────────────────────
    lockzhiner_vision_module::edit::Edit edit;
    if (use_display)
    {
        if (!edit.StartAndAcceptConnection())
        {
            std::cerr << "Error: Failed to start and accept connection." << std::endl;
            return EXIT_FAILURE;
        }
        std::cout << "Display enabled, device connected successfully." << std::endl;
    }

    // ── 模型初始化 ──────────────────────────────────────────────────────────
    lockzhiner_vision_module::vision::PaddleDet model;
    if (!model.Initialize(argv[1]))
    {
        std::cerr << "Error: Failed to initialize model." << std::endl;
        return 1;
    }
    model.SetThreshold(0.7, 0.3);
    std::cout << "Model initialized successfully." << std::endl;

    // ── 摄像头初始化 ─────────────────────────────────────────────────────────
    cv::VideoCapture cap;
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 320);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 240);
    cap.open(0);
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open camera." << std::endl;
        return 1;
    }

    cv::Mat input_mat;
    TargetTracker tracker;
    high_resolution_clock::time_point last_time = high_resolution_clock::now();

    while (true)
    {
        // ── 采集一帧 ─────────────────────────────────────────────────────────
        cap >> input_mat;
        if (input_mat.empty())
        {
            std::cerr << "Warning: Captured an empty frame." << std::endl;
            continue;
        }

        // 计算时间间隔 dt
        high_resolution_clock::now();
        high_resolution_clock::time_point current_time = high_resolution_clock::now();
        float dt = duration_cast<milliseconds>(current_time - last_time).count() / 1000.0f;
        last_time = current_time;

        // ── 目标检测 ─────────────────────────────────────────────────────────
        high_resolution_clock::time_point t0 = high_resolution_clock::now();
        auto det_results = model.Predict(input_mat);
        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        std::cout << "Inference: " << duration_cast<milliseconds>(t1 - t0).count() << " ms"
                  << "  detections: " << det_results.size() << std::endl;

        // 预测当前位置
        cv::Point2f predicted_pt = tracker.Predict(dt);

        // 初始化本帧状态信息，用于 UI 右上角固定显示
        std::string status_text = "B:0,0,0";
        cv::Scalar status_color = cv::Scalar(0, 0, 255); // 默认红色（未识别）

        // 如果模型没有识别到任何目标，更新追踪器并下发标志位 0
        if (det_results.empty())
        {
            tracker.MarkLost();
            usart.Write("B:0,0,0\r\n");
        }

        cv::Mat result_image;
        if (use_display)
        {
            result_image = input_mat.clone();
        }

        for (const auto &det : det_results)
        {
            // ── 将检测框裁剪到图像范围内 ──────────────────────────────────────
            cv::Rect bbox = det.box & cv::Rect(0, 0, input_mat.cols, input_mat.rows);
            if (bbox.empty())
                continue;

            if (use_display)
            {
                // 绘制检测框（蓝色）
                cv::rectangle(result_image, bbox, cv::Scalar(255, 128, 0), 1);
            }

            // ── 在 ROI 内进行四边形轮廓检测 ───────────────────────────────────
            cv::Mat roi = input_mat(bbox);

            cv::Mat gray, edges;
            cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
            cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);
            cv::Canny(gray, edges, 50, 150);

            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

            // 选取面积最大的四边形
            double best_area = 0.0;
            std::vector<cv::Point> best_approx;
            cv::Point2f best_center(-1.0f, -1.0f);

            for (const auto &contour : contours)
            {
                double area = cv::contourArea(contour);
                if (area < 200.0)
                    continue;

                std::vector<cv::Point> approx;
                double epsilon = cv::arcLength(contour, true) * 0.03;
                cv::approxPolyDP(contour, approx, epsilon, true);

                if (approx.size() == 4 && area > best_area)
                {
                    best_area = area;
                    best_approx = approx;

                    // 使用对角线交点计算四边形中心（ROI 坐标系）
                    // 在透视投影下，矩形的几何中心即为对角线的交点，这比直接平均四个顶点更准确
                    cv::Point2f p0 = static_cast<cv::Point2f>(approx[0]);
                    cv::Point2f p1 = static_cast<cv::Point2f>(approx[1]);
                    cv::Point2f p2 = static_cast<cv::Point2f>(approx[2]);
                    cv::Point2f p3 = static_cast<cv::Point2f>(approx[3]);
                    cv::Point2f c = GetDiagonalIntersection(p0, p2, p1, p3);

                    // 转换为原图坐标系
                    best_center.x = c.x + static_cast<float>(bbox.x);
                    best_center.y = c.y + static_cast<float>(bbox.y);
                }
            }

            // ── 串口下发中心坐标 ──────────────────────────────────────────────
            cv::Point2f final_center = best_center;
            int find_flag = 1; // 模型识别标志位 (1: 识别到目标)

            if (best_center.x < 0.0f)
            {
                // 如果矩形检测没成功，使用模型检测框中心作为 fallback
                final_center.x = bbox.x + bbox.width / 2.0f;
                final_center.y = bbox.y + bbox.height / 2.0f;
            }

            // --- 卡尔曼滤波更新 ---
            tracker.Update(final_center);
            // 使用滤波后的状态，并超前预测 0.03s (30ms) 以抵消视觉/串口/执行延迟
            cv::Point2f filtered_pt = tracker.Predict(0.00f); 

            // 协议格式：B:x,y,flag\r\n
            // 串口下发的是数值标志位 (1)，保持下位机解析稳定
            std::string msg = "B:" +
                              std::to_string(static_cast<int>(filtered_pt.x)) + "," +
                              std::to_string(static_cast<int>(filtered_pt.y)) + "," +
                              std::to_string(find_flag) + "\r\n";
            usart.Write(msg.c_str());

            // 更新 UI 显示用的状态文本和颜色
            status_text = "B:" + std::to_string(static_cast<int>(filtered_pt.x)) + "," +
                          std::to_string(static_cast<int>(filtered_pt.y)) + "," +
                          std::to_string(find_flag);
            status_color = (best_center.x >= 0.0f) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255);

            // ── 可视化 ────────────────────────────────────────────────────────
            if (use_display)
            {
                if (best_center.x >= 0.0f)
                {
                    // 只有识别到矩形时才绘制绿色轮廓和角点
                    std::vector<cv::Point> approx_global(best_approx.size());
                    for (size_t j = 0; j < best_approx.size(); j++)
                    {
                        approx_global[j].x = best_approx[j].x + bbox.x;
                        approx_global[j].y = best_approx[j].y + bbox.y;
                    }
                    cv::drawContours(result_image,
                                     std::vector<std::vector<cv::Point>>{approx_global},
                                     -1, cv::Scalar(0, 255, 0), 1);
                    for (const auto &pt : approx_global)
                    {
                        cv::circle(result_image, pt, 2, cv::Scalar(0, 0, 255), -1);
                    }
                }

                // 绘制原始检测中心（红色实心小点）
                cv::circle(result_image, final_center, 2, cv::Scalar(0, 0, 255), -1);

                // 绘制滤波/预测后的几何中心（青色空心圆环）
                cv::circle(result_image, filtered_pt, 6, cv::Scalar(255, 255, 0), 1);
                // 在圆环中心点一下（青色极小点，用于定位）
                cv::circle(result_image, filtered_pt, 1, cv::Scalar(255, 255, 0), -1);

                // 在目标点上方也保留一个简短的提示
                std::string point_label = "(" + std::to_string(static_cast<int>(filtered_pt.x)) + "," + std::to_string(static_cast<int>(filtered_pt.y)) + ")";
                cv::putText(result_image, point_label,
                            cv::Point(static_cast<int>(filtered_pt.x) + 10, static_cast<int>(filtered_pt.y) - 10),
                            cv::FONT_HERSHEY_SIMPLEX, 0.35, status_color, 1);
            }
        }

        // ── 显示结果 ─────────────────────────────────────────────────────────
        if (use_display)
        {
            // 在右上角固定显示协议文本，不带 \r\n
            int baseline = 0;
            cv::Size text_size = cv::getTextSize(status_text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            cv::Point text_org(result_image.cols - text_size.width - 10, text_size.height + 15);

            // 绘制文本背景框，增强可读性
            cv::rectangle(result_image, text_org + cv::Point(-5, 5), text_org + cv::Point(text_size.width + 5, -text_size.height - 5), cv::Scalar(0, 0, 0), -1);
            cv::putText(result_image, status_text, text_org, cv::FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1);

            edit.Print(result_image);
        }
    }

    cap.release();
    return 0;
}