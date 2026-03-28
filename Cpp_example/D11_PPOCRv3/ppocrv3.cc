#include <ncnn/net.h>
#include <lockzhiner_vision_module/edit/edit.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <algorithm>

using namespace std;
using namespace std::chrono;

// OCR配置参数
const cv::Size OCR_INPUT_SIZE(320, 48);

const bool USE_SPACE_CHAR = true;
const float MEAN_VALS[3] = {127.5f, 127.5f, 127.5f};
const float NORM_VALS[3] = {1.0f/127.5f, 1.0f/127.5f, 1.0f/127.5f};

ncnn::Net ocr_net;


vector<string> char_list;  // 从字典加载的字符列表

// 加载字符字典
vector<string> load_character_dict(const string& path, bool use_space_char) {
    vector<string> temp_list;
    ifstream file(path);
    if (!file.is_open()) {
        cerr << "Failed to open dictionary file: " << path << endl;
        return {};
    }
    
    string line;    
    while (getline(file, line)) {
        line.erase(remove(line.begin(), line.end(), '\r'), line.end());
        line.erase(remove(line.begin(), line.end(), '\n'), line.end());
        temp_list.push_back(line);
    }
    
    if (use_space_char) {
        temp_list.push_back(" ");
    }
    
    vector<string> char_list{"blank"};
    char_list.insert(char_list.end(), temp_list.begin(), temp_list.end());
    return char_list;
}

// CTC解码
string decode_ctc(const ncnn::Mat& out) {
    vector<int> indices;
    const int num_timesteps = out.h;
    const int num_classes = out.w;

    for (int t = 0; t < num_timesteps; ++t) {
        const float* prob = out.row(t);
        int max_idx = 0;
        float max_prob = prob[0];
        
        for (int c = 0; c < num_classes; ++c) {
            if (prob[c] > max_prob) {
                max_idx = c;
                max_prob = prob[c];
            }
        }
        indices.push_back(max_idx);
    }

    string result;
    int prev_idx = -1;
    
    for (int idx : indices) {
        if (idx == 0) {
            prev_idx = -1;
            continue;
        }
        if (idx != prev_idx) {
            if (idx < char_list.size()) {
                result += char_list[idx];
            }
            prev_idx = idx;
        }
    }
    
    return result;
}


// 初始化OCR模型
bool InitOCRModel(const string& param_path, const string& model_path, const string& dict_path) {
    if (!ocr_net.load_param(param_path.c_str()) && !ocr_net.load_model(model_path.c_str())) {
        char_list = load_character_dict(dict_path, USE_SPACE_CHAR);
        return !char_list.empty();
    }
    return false;
}

// 文字识别
string RecognizePlate(cv::Mat plate_img) {
    // 图像预处理
    cv::resize(plate_img, plate_img, OCR_INPUT_SIZE);
    ncnn::Mat in = ncnn::Mat::from_pixels(plate_img.data, 
                                        ncnn::Mat::PIXEL_BGR,
                                        plate_img.cols, 
                                        plate_img.rows);
    // PP-OCR风格归一化
    in.substract_mean_normalize(MEAN_VALS, NORM_VALS);

    // 模型推理
    ncnn::Extractor ex = ocr_net.create_extractor();
    ex.input("in0", in);
    
    ncnn::Mat out;
    ex.extract("out0", out);
    
    // CTC解码
    string license = decode_ctc(out);
    return license;
}

int main(int argc, char** argv) {
    // 参数验证
    if (argc != 5) {
        cerr << "Usage: " << argv[0] 
             << " <ocr_param> <ocr_model> <dict_path> [image_path]\n"
             << "Example:\n"
             << "  Realtime: " << argv[0] << " ocr.param ocr.bin ppocr_keys_v1.txt\n"
             << "  Image:    " << argv[0] << " ocr.param ocr.bin ppocr_keys_v1.txt test.jpg\n";
        return 1;
    }
    // 初始化OCR模型和字典
    if (!InitOCRModel(argv[1], argv[2], argv[3])) {
        cerr << "Failed to initialize OCR system" << endl;
        return 1;
    }

    // 图片处理模式
    cv::Mat image = cv::imread(argv[4]);
    if (image.empty()) {
        cerr << "Failed to read image: " << argv[4] << endl;
        return 1;
    }

    auto ocr_start = std::chrono::high_resolution_clock::now();
    string result = RecognizePlate(image);
    auto ocr_end = std::chrono::high_resolution_clock::now();
    std::cout << "OCR: " << std::chrono::duration<double>(ocr_end - ocr_start).count() << "s\n";

    cout << "  识别结果: " << result << endl;
    cv::waitKey(0);
}