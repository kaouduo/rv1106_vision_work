#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

/**
 * @brief OCR检测结果结构体定义
 */
typedef struct rknn_point_t
{
    int x; /**< X坐标 */
    int y; /**< Y坐标 */
} rknn_point_t;

/**
 * @brief 四边形文本框结构体，表示一个文本区域的四个顶点及其置信度
 */
typedef struct rknn_quad_t
{
    rknn_point_t left_top;     /**< 左上角点 */
    rknn_point_t right_top;    /**< 右上角点 */
    rknn_point_t right_bottom; /**< 右下角点 */
    rknn_point_t left_bottom;  /**< 左下角点 */
    float score;               /**< 检测框的置信度分数 */
} rknn_quad_t;

/**
 * @brief OCR检测结果结构体，包含多个四边形文本框及数量
 */
typedef struct
{
    rknn_quad_t box[1000]; /**< 最多支持1000个文本框 */
    int count;             /**< 实际检测到的文本框数量 */
} ppocr_det_result;

/**
 * @brief DBNet后处理函数，将模型输出转换为原始图像上的文本检测框
 *
 * @param output 模型输出数据指针
 * @param det_out_w 模型输出宽度
 * @param det_out_h 模型输出高度
 * @param db_threshold 二值化阈值
 * @param db_box_threshold 检测框过滤阈值
 * @param use_dilation 是否使用膨胀操作优化文本区域
 * @param db_score_mode 分数计算模式 ("slow" 或 "fast")
 * @param db_unclip_ratio 控制文本框扩展的比例
 * @param db_box_type 输出框类型 ("quad" 表示四边形)
 * @param scale_w 宽度方向缩放比例（原始图 / 输入图）
 * @param scale_h 高度方向缩放比例（原始图 / 输入图）
 * @param results 输出参数，用于保存最终检测结果
 * @param debug 是否启用调试输出，默认关闭
 * @return 返回错误码（0表示成功）
 */
int dbnet_postprocess(float *output, int det_out_w, int det_out_h,
                      float db_threshold, float db_box_threshold,
                      bool use_dilation, const std::string &db_score_mode,
                      float db_unclip_ratio, const std::string &db_box_type,
                      float scale_w, float scale_h,
                      ppocr_det_result *results,
                      bool debug = false);

/**
 * @brief 获取最小外接矩形的四个顶点（按顺序排列）
 *
 * @param box OpenCV旋转矩形对象
 * @param ssid 用于返回该矩形的面积
 * @return 返回包含四个顶点坐标的二维向量
 */
std::vector<std::vector<float>> GetMiniBoxes(cv::RotatedRect box, float &ssid);

/**
 * @brief 浮点数限制函数，确保数值在[min, max]范围内
 *
 * @param x 输入浮点数
 * @param min 下限
 * @param max 上限
 * @return 限制后的结果
 */
float clampf(float x, float min, float max);

/**
 * @brief 整数限制函数，确保数值在[min, max]范围内
 *
 * @param x 输入整数
 * @param min 下限
 * @param max 上限
 * @return 限制后的结果
 */
int clamp(int x, int min, int max);

/**
 * @brief 计算多边形区域内预测图的平均得分（用于文本区域评分）
 *
 * @param contour 多边形轮廓点集
 * @param pred 预测图（cv::Mat格式）
 * @return 返回得分
 */
float PolygonScoreAcc(std::vector<cv::Point> contour, cv::Mat pred);

/**
 * @brief 快速计算指定四边形区域内的平均得分
 *
 * @param box_array 四边形顶点数组
 * @param pred 预测图（cv::Mat格式）
 * @return 返回得分
 */
float BoxScoreFast(std::vector<std::vector<float>> box_array, cv::Mat pred);

/**
 * @brief 将输入的四边形框进行“膨胀”扩展，扩大文本区域范围
 *
 * @param box 输入四边形顶点数组
 * @param unclip_ratio 扩展比例
 * @return 返回扩展后的旋转矩形
 */
cv::RotatedRect UnClip(std::vector<std::vector<float>> &box, const float &unclip_ratio);

/**
 * @brief 将输入的四个点按照顺时针方向排序
 *
 * @param pts 输入的四个点坐标
 * @return 返回排序后的点集合
 */
std::vector<std::vector<int>> OrderPointsClockwise(std::vector<std::vector<int>> pts);

/**
 * @brief 对输入图像进行预处理，使其适配模型输入要求。
 *
 * 该函数完成以下操作：
 * - 调整图像尺寸至模型输入大小
 * - 将像素值归一化为浮点数范围 [0, 1]
 * - 根据量化参数进行量化处理（支持 INT8 量化模型）
 * - 将数据格式从 HWC 转换为 CHW（通道优先）
 * - 返回处理后的 cv::Mat 数据
 *
 * @param image 输入原始图像（OpenCV Mat 格式，BGR 或灰度图）
 * @param input_dims 模型输入张量的维度，如 {N, C, H, W}，用于获取目标高度和宽度
 * @param input_scale 量化缩放因子（scale），用于将浮点数转换为整数（适用于 INT8 量化模型）
 * @param input_zp 量化零点（zero point），用于偏移量调整（适用于 INT8 量化模型）
 *
 * @return 返回预处理后的图像数据，格式为 CV_32FC3 或 CV_32FC1，通道顺序为 CHW
 */
cv::Mat preprocess(const cv::Mat &image,
                   const std::vector<size_t> &input_dims,
                   float input_scale,
                   int input_zp);

/**
 * @brief 在图像上绘制检测结果的边界框。
 *
 * 该函数根据 OCR 检测结果，在输入图像上绘制出对应的边界框，便于可视化模型输出。
 *
 * @param img 输入图像的指针（OpenCV Mat 格式），将在该图像上绘制边界框
 * @param results OCR 检测结果结构体，包含多个边界框及其坐标信息
 * @param thickness 绘制边界框的线条粗细，默认为 2
 * @param debug 是否启用调试模式，默认为 false。若为 true，可能显示额外调试信息或更详细的框选效果
 */
void draw_boxes(cv::Mat *img, const ppocr_det_result &results, int thickness = 2, bool debug = false);
#endif // POSTPROCESS_H