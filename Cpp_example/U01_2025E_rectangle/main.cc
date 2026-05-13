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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <lockzhiner_vision_module/edit/edit.h>
#include <lockzhiner_vision_module/periphery/usart/usart.h>
#include <lockzhiner_vision_module/vision/deep_learning/detection/paddle_det.h>
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>

using namespace std::chrono;


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
        return cv::Point2f((x1 + x2 + x3 + x4) / 4.0f, (y1 + y2 + y3 + y4) / 4.0f);
    }

    float x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator;
    float y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator;

    return cv::Point2f(x, y);
}

float DistancePointToLine(cv::Point2f p, cv::Point2f a, cv::Point2f b)
{
    cv::Point2f ab = b - a;
    cv::Point2f ap = p - a;
    float denom = std::sqrt(ab.x * ab.x + ab.y * ab.y);
    if (denom < 1e-6f)
        return std::numeric_limits<float>::max();
    return std::abs(ab.x * ap.y - ab.y * ap.x) / denom;
}

float EstimateInnerRadiusFromQuad(const std::vector<cv::Point> &quad, cv::Point2f center)
{
    if (quad.size() != 4)
        return -1.0f;

    float min_dist = std::numeric_limits<float>::max();
    for (int i = 0; i < 4; ++i)
    {
        cv::Point2f a = static_cast<cv::Point2f>(quad[i]);
        cv::Point2f b = static_cast<cv::Point2f>(quad[(i + 1) % 4]);
        min_dist = std::min(min_dist, DistancePointToLine(center, a, b));
    }

    return min_dist;
}

std::vector<cv::Point2f> OrderQuadPointsClockwise(const std::vector<cv::Point> &quad)
{
    if (quad.size() != 4)
        return {};

    std::vector<cv::Point2f> pts(4);
    for (int i = 0; i < 4; ++i)
        pts[i] = static_cast<cv::Point2f>(quad[i]);

    int tl = 0, tr = 0, br = 0, bl = 0;
    float min_sum = std::numeric_limits<float>::max();
    float max_sum = -std::numeric_limits<float>::max();
    float min_diff = std::numeric_limits<float>::max();
    float max_diff = -std::numeric_limits<float>::max();

    for (int i = 0; i < 4; ++i)
    {
        float s = pts[i].x + pts[i].y;
        float d = pts[i].x - pts[i].y;
        if (s < min_sum)
        {
            min_sum = s;
            tl = i;
        }
        if (s > max_sum)
        {
            max_sum = s;
            br = i;
        }
        if (d < min_diff)
        {
            min_diff = d;
            tr = i;
        }
        if (d > max_diff)
        {
            max_diff = d;
            bl = i;
        }
    }

    return {pts[tl], pts[tr], pts[br], pts[bl]};
}

bool ComputeQuadHomographyToRect(const std::vector<cv::Point> &quad, cv::Mat &H, cv::Mat &Hinv, cv::Size2f &dst_size)
{
    auto src = OrderQuadPointsClockwise(quad);
    if (src.size() != 4)
        return false;

    float width = std::max(cv::norm(src[1] - src[0]), cv::norm(src[2] - src[3]));
    float height = std::max(cv::norm(src[3] - src[0]), cv::norm(src[2] - src[1]));
    if (width < 2.0f || height < 2.0f)
        return false;

    dst_size = cv::Size2f(width, height);

    std::vector<cv::Point2f> dst = {cv::Point2f(0.0f, 0.0f), cv::Point2f(width - 1.0f, 0.0f),
                                    cv::Point2f(width - 1.0f, height - 1.0f), cv::Point2f(0.0f, height - 1.0f)};

    H = cv::getPerspectiveTransform(src, dst);
    if (H.empty())
        return false;

    Hinv = H.inv();
    return !Hinv.empty();
}

cv::Point2f ClampPointToImage(cv::Point2f p, const cv::Mat &img)
{
    if (img.empty())
        return p;

    p.x = std::clamp(p.x, 0.0f, static_cast<float>(img.cols - 1));
    p.y = std::clamp(p.y, 0.0f, static_cast<float>(img.rows - 1));
    return p;
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
    model.SetThreshold(0.65, 0.3);
    std::cout << "Model initialized successfully." << std::endl;

    // ── 摄像头初始化 ─────────────────────────────────────────────────────────
    cv::VideoCapture cap;
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 320);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 320);
    cap.open(0);
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open camera." << std::endl;
        return 1;
    }

    cv::Mat input_mat;
    int last_sent_x = 0;
    int last_sent_y = 0;
    int last_sent_circle_x = 0;
    int last_sent_circle_y = 0;
    int last_sent_area = 0;

    // ── FPS 统计变量 ────────────────────────────────────────────────────────
    int fps_count = 0;
    float current_fps = 0.0f;
    high_resolution_clock::time_point fps_start_time = high_resolution_clock::now();

    bool circle_anim_active = false;
    double circle_anim_phase = 0.0;
    high_resolution_clock::time_point circle_anim_last_update_time = high_resolution_clock::now();
    high_resolution_clock::time_point circle_anim_last_print_time = high_resolution_clock::now();

    while (true)
    {
        // ── 采集一帧 ─────────────────────────────────────────────────────────
        cap >> input_mat;
        if (input_mat.empty())
        {
            std::cerr << "Warning: Captured an empty frame." << std::endl;
            continue;
        }

        high_resolution_clock::time_point current_time = high_resolution_clock::now();

        // 计算 FPS（每 1 秒更新一次）
        fps_count++;
        float fps_elapsed = duration_cast<milliseconds>(current_time - fps_start_time).count() / 1000.0f;
        if (fps_elapsed >= 1.0f)
        {
            current_fps = fps_count / fps_elapsed;
            fps_count = 0;
            fps_start_time = current_time;
        }

        // ── 目标检测 ─────────────────────────────────────────────────────────
        high_resolution_clock::time_point t0 = high_resolution_clock::now();
        auto det_results = model.Predict(input_mat);
        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        long inference_ms = duration_cast<milliseconds>(t1 - t0).count();

        // 初始化本帧状态信息，用于 UI 右上角固定显示
        std::string status_text = "B:0,0,0,0";
        cv::Scalar status_color = cv::Scalar(0, 0, 255);
        bool has_valid_rectangle = false;

        // 如果模型没有识别到任何目标，更新追踪器并下发标志位 0，面积 0
        if (det_results.empty())
        {
            std::cout << "NPU: " << inference_ms << " ms | Traditional: 0 ms | detections: 0 | FPS: " << std::fixed
                      << std::setprecision(1) << current_fps << std::endl;
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
                cv::rectangle(result_image, bbox, cv::Scalar(0, 0, 255), 1);
                std::string conf_text = cv::format("conf:%.2f", det.score);
                cv::Point conf_org(bbox.x, std::max(15, bbox.y - 5));
                cv::putText(result_image, conf_text, conf_org, cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 255),
                            1);
            }

            // ── 在 ROI 内进行四边形轮廓检测 ───────────────────────────────────
            high_resolution_clock::time_point t_trad_0 = high_resolution_clock::now();
            cv::Mat roi = input_mat(bbox);

            cv::Mat gray, edges;
            cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
            cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);
            cv::Canny(gray, edges, 50, 150);

            std::vector<std::vector<cv::Point>> contours;
            std::vector<cv::Vec4i> hierarchy;
            cv::findContours(edges, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

            double best_area = 0.0;
            double best_width = 0.0;
            std::vector<cv::Point> best_approx;
            cv::Point2f best_center(-1.0f, -1.0f);
            constexpr double min_area_ratio = 0.005;

            for (int i = 0; i < static_cast<int>(contours.size()); ++i)
            {
                if (hierarchy[i][3] != -1)
                    continue;
                int child = hierarchy[i][2];

                double parent_area = cv::contourArea(contours[i]);
                if (parent_area < 50.0)
                    continue;

                double max_child_area = 0.0;
                for (int c = child; c != -1; c = hierarchy[c][0])
                {
                    max_child_area = std::max(max_child_area, cv::contourArea(contours[c]));
                }

                double area_ratio = (child == -1) ? 1.0 : (max_child_area / parent_area);
                if (area_ratio < min_area_ratio)
                    continue;

                std::vector<cv::Point> approx;
                double epsilon = cv::arcLength(contours[i], true) * 0.04;
                cv::approxPolyDP(contours[i], approx, epsilon, true);
                if (approx.size() != 4)
                    continue;

                cv::Rect rect = cv::boundingRect(approx);
                if (rect.width > best_width)
                {
                    best_width = static_cast<double>(rect.width);
                    best_area = parent_area;
                    best_approx = approx;
                    cv::Point2f p0 = static_cast<cv::Point2f>(approx[0]);
                    cv::Point2f p1 = static_cast<cv::Point2f>(approx[1]);
                    cv::Point2f p2 = static_cast<cv::Point2f>(approx[2]);
                    cv::Point2f p3 = static_cast<cv::Point2f>(approx[3]);
                    cv::Point2f c = GetDiagonalIntersection(p0, p2, p1, p3);
                    best_center.x = c.x + static_cast<float>(bbox.x);
                    best_center.y = c.y + static_cast<float>(bbox.y);
                }
            }
            high_resolution_clock::time_point t_trad_1 = high_resolution_clock::now();
            long trad_ms = duration_cast<milliseconds>(t_trad_1 - t_trad_0).count();
            std::cout << "NPU: " << inference_ms << " ms | Traditional: " << trad_ms
                      << " ms | detections: " << det_results.size() << " | FPS: " << std::fixed << std::setprecision(1)
                      << current_fps << std::endl;

            // ── 串口下发中心坐标 ──────────────────────────────────────────────
            cv::Point2f final_center = best_center;
            double final_area = best_area;
            int find_flag = 1; // 模型识别标志位 (1: 识别到目标)

            if (best_center.x < 0.0f)
            {
                continue;
            }

            cv::Point2f filtered_pt = final_center;
            has_valid_rectangle = true;

            cv::Point2f serial_pt = filtered_pt;

            if (best_approx.size() == 4)
            {
                std::vector<cv::Point> approx_global(best_approx.size());
                for (size_t j = 0; j < best_approx.size(); j++)
                {
                    approx_global[j].x = best_approx[j].x + bbox.x;
                    approx_global[j].y = best_approx[j].y + bbox.y;
                }

                constexpr int kTotalRingCount = 5;
                constexpr int kTargetRingIndex = 3;
                constexpr double kCirclePeriodSec = 18.0;

                cv::Point2f circle_pt = serial_pt;
                bool has_circle_pt = false;

                cv::Mat H, Hinv;
                cv::Size2f dst_size;
                if (ComputeQuadHomographyToRect(approx_global, H, Hinv, dst_size))
                {
                    std::vector<cv::Point2f> center_in = {final_center};
                    std::vector<cv::Point2f> center_out;
                    cv::perspectiveTransform(center_in, center_out, H);

                    if (!center_out.empty())
                    {
                        cv::Point2f c = center_out[0];
                        float inner_radius = std::min(std::min(c.x, (dst_size.width - 1.0f) - c.x),
                                                      std::min(c.y, (dst_size.height - 1.0f) - c.y));

                        if (inner_radius > 1.0f)
                        {
                            float ring_ratio =
                                static_cast<float>(kTargetRingIndex) / static_cast<float>(kTotalRingCount);
                            float ring_radius = inner_radius * ring_ratio;

                            if (ring_radius > 1.0f)
                            {
                                if (!circle_anim_active)
                                {
                                    circle_anim_active = true;
                                    circle_anim_last_update_time = current_time;
                                    circle_anim_last_print_time = current_time - milliseconds(1000);
                                }

                                double dt_sec =
                                    duration_cast<milliseconds>(current_time - circle_anim_last_update_time).count() /
                                    1000.0;
                                if (dt_sec < 0.0)
                                    dt_sec = 0.0;
                                circle_anim_phase = std::fmod(circle_anim_phase + dt_sec / kCirclePeriodSec, 1.0);
                                circle_anim_last_update_time = current_time;

                                double angle = -CV_PI * 0.5 + 2.0 * CV_PI * circle_anim_phase;
                                cv::Point2f circle_pt_warp(static_cast<float>(c.x + ring_radius * std::cos(angle)),
                                                           static_cast<float>(c.y + ring_radius * std::sin(angle)));

                                std::vector<cv::Point2f> pt_in = {circle_pt_warp};
                                std::vector<cv::Point2f> pt_out;
                                cv::perspectiveTransform(pt_in, pt_out, Hinv);
                                if (!pt_out.empty())
                                {
                                    circle_pt = ClampPointToImage(pt_out[0], input_mat);
                                    has_circle_pt = true;
                                }
                            }
                        }
                    }
                }

                if (!has_circle_pt)
                {
                    float inner_radius = EstimateInnerRadiusFromQuad(approx_global, final_center);
                    if (inner_radius > 1.0f)
                    {
                        float ring_ratio = static_cast<float>(kTargetRingIndex) / static_cast<float>(kTotalRingCount);
                        float ring_radius = inner_radius * ring_ratio;
                        if (ring_radius > 1.0f)
                        {
                            if (!circle_anim_active)
                            {
                                circle_anim_active = true;
                                circle_anim_last_update_time = current_time;
                                circle_anim_last_print_time = current_time - milliseconds(1000);
                            }

                            double dt_sec =
                                duration_cast<milliseconds>(current_time - circle_anim_last_update_time).count() /
                                1000.0;
                            if (dt_sec < 0.0)
                                dt_sec = 0.0;
                            circle_anim_phase = std::fmod(circle_anim_phase + dt_sec / kCirclePeriodSec, 1.0);
                            circle_anim_last_update_time = current_time;

                            double angle = -CV_PI * 0.5 + 2.0 * CV_PI * circle_anim_phase;

                            circle_pt = cv::Point2f(static_cast<float>(final_center.x + ring_radius * std::cos(angle)),
                                                    static_cast<float>(final_center.y + ring_radius * std::sin(angle)));
                            circle_pt = ClampPointToImage(circle_pt, input_mat);
                            has_circle_pt = true;
                        }
                    }
                }

                if (has_circle_pt)
                {
                    serial_pt = circle_pt;
                    if (use_display)
                    {
                        cv::circle(result_image, circle_pt, 2, cv::Scalar(255, 0, 0), -1);
                    }

                    if (duration_cast<milliseconds>(current_time - circle_anim_last_print_time).count() >= 200)
                    {
                        circle_anim_last_print_time = current_time;
                        std::cout << "P:" << static_cast<int>(std::round(circle_pt.x)) << ","
                                  << static_cast<int>(std::round(circle_pt.y)) << std::endl;
                    }
                }
            }

            // 协议格式：B:center_x,center_y,flag,area,circle_x,circle_y\r\n
            // 串口下发的是数值标志位 (1)，面积取整数
            last_sent_x = static_cast<int>(std::round(filtered_pt.x));
            last_sent_y = static_cast<int>(std::round(filtered_pt.y));
            last_sent_circle_x = static_cast<int>(std::round(serial_pt.x));
            last_sent_circle_y = static_cast<int>(std::round(serial_pt.y));
            last_sent_area = static_cast<int>(final_area);
            std::string msg = "B:" + std::to_string(last_sent_x) + "," + std::to_string(last_sent_y) + "," +
                              std::to_string(find_flag) + "," + std::to_string(last_sent_area) + "," +
                              std::to_string(last_sent_circle_x) + "," + std::to_string(last_sent_circle_y) + "\r\n";
            usart.Write(msg.c_str());

            // 更新 UI 显示用的状态文本和颜色
            status_text = "B:" + std::to_string(last_sent_x) + "," + std::to_string(last_sent_y) + "," +
                          std::to_string(find_flag) + "," + std::to_string(static_cast<int>(final_area)) + "," +
                          std::to_string(last_sent_circle_x) + "," + std::to_string(last_sent_circle_y);
            status_color = (best_center.x >= 0.0f) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255);

            // ── 可视化 ────────────────────────────────────────────────────────
            if (use_display)
            {
                if (best_center.x >= 0.0f)
                {
                    std::vector<cv::Point> approx_global(best_approx.size());
                    for (size_t j = 0; j < best_approx.size(); j++)
                    {
                        approx_global[j].x = best_approx[j].x + bbox.x;
                        approx_global[j].y = best_approx[j].y + bbox.y;
                    }
                    cv::drawContours(result_image, std::vector<std::vector<cv::Point>>{approx_global}, -1,
                                     cv::Scalar(0, 255, 0), 1);
                    for (const auto &pt : approx_global)
                    {
                        cv::circle(result_image, pt, 2, cv::Scalar(0, 0, 255), -1);
                    }
                }

                cv::circle(result_image, final_center, 2, cv::Scalar(0, 0, 255), -1);

                // 绘制滤波/预测后的几何中心（青色空心圆环）
                cv::circle(result_image, filtered_pt, 6, cv::Scalar(255, 255, 0), 1);
                // 在圆环中心点一下（青色极小点，用于定位）
                cv::circle(result_image, filtered_pt, 1, cv::Scalar(255, 255, 0), -1);

                // 在目标点上方也保留一个简短的提示
                std::string point_label = "(" + std::to_string(static_cast<int>(filtered_pt.x)) + "," +
                                          std::to_string(static_cast<int>(filtered_pt.y)) + ")";
                cv::putText(result_image, point_label,
                            cv::Point(static_cast<int>(filtered_pt.x) + 10, static_cast<int>(filtered_pt.y) - 10),
                            cv::FONT_HERSHEY_SIMPLEX, 0.35, status_color, 1);
            }
        }

        if (!has_valid_rectangle)
        {
            circle_anim_active = false;
            last_sent_x = 0;
            last_sent_y = 0;
            last_sent_circle_x = 0;
            last_sent_circle_y = 0;
            last_sent_area = 0;
            std::string lost_msg = "B:0,0,0,0,0,0\r\n";
            usart.Write(lost_msg.c_str());
            status_text = "B:0,0,0,0,0,0";
            status_color = cv::Scalar(0, 0, 255);
        }

        // ── 显示结果 ─────────────────────────────────────────────────────────
        if (use_display)
        {
            if (has_valid_rectangle)
            {
                int baseline = 0;
                cv::Size text_size = cv::getTextSize(status_text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
                cv::Point text_org(result_image.cols - text_size.width - 10, text_size.height + 15);
                cv::rectangle(result_image, text_org + cv::Point(-5, 5),
                              text_org + cv::Point(text_size.width + 5, -text_size.height - 5), cv::Scalar(0, 0, 0),
                              -1);
                cv::putText(result_image, status_text, text_org, cv::FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1);
            }
            edit.Print(result_image);
        }
    }

    cap.release();
    return 0;
}