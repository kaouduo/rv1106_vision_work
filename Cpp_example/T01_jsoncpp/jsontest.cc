#include "json/json.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>

// 打印帮助信息
void printUsage(const std::string& progName) {
    std::cout << "Usage: " << progName << " [option]\n";
    std::cout << "Options:\n";
    std::cout << "  write      - Write sample JSON to stdout\n";
    std::cout << "  writestr   - Write JSON to string\n";
    std::cout << "  readstr    - Read JSON from string\n";
    std::cout << "  writefile  - Write JSON to file (example.json)\n";
    std::cout << "  readfile   - Read JSON from file (example.json)\n";
}

// 将 Json::Value 转换为字符串输出
std::string writeToJsonString(const Json::Value& root) {
    Json::StreamWriterBuilder builder;
    std::ostringstream oss;
    std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
    writer->write(root, &oss);
    return oss.str();
}

// 构建一个测试用的 JSON 对象
Json::Value createSampleJson() {
    Json::Value root;
    root["Name"] = "robin";
    root["Age"] = 20;

    Json::Value hobbies(Json::arrayValue);
    hobbies.append("reading");
    hobbies.append("coding");
    hobbies.append("hiking");

    root["Hobbies"] = hobbies;

    Json::Value address;
    address["City"] = "Beijing";
    address["ZipCode"] = "100000";
    root["Address"] = address;

    return root;
}

// 从字符串解析 JSON
bool parseJsonFromString(const std::string& jsonStr, Json::Value& root) {
    Json::CharReaderBuilder builder;
    std::string errs;
    std::istringstream jsonStream(jsonStr);
    bool ok = Json::parseFromStream(builder, jsonStream, &root, nullptr);
    if (!ok) {
        std::cerr << "Failed to parse JSON string." << std::endl;
        return false;
    }
    return true;
}

// 写入 JSON 到文件
bool writeJsonToFile(const std::string& filename, const Json::Value& root) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return false;
    }

    Json::StreamWriterBuilder builder;
    std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
    writer->write(root, &ofs);
    ofs.close();
    return true;
}

// 从文件读取 JSON
bool readJsonFromFile(const std::string& filename, Json::Value& root) {
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
        return false;
    }

    // 使用 JsonCpp 提供的 parseFromStream 方法更简洁
    Json::CharReaderBuilder builder;
    std::string errs;
    bool ok = Json::parseFromStream(builder, ifs, &root, nullptr);
    ifs.close();

    if (!ok) {
        std::cerr << "Failed to parse JSON from file." << std::endl;
        return false;
    }

    return true;
}

int main(int argc, char* argv[]) {
    std::string operation = "write"; // 默认操作
    if (argc > 1) {
        operation = argv[1];
    } else {
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }

    if (operation == "write") {
        // 直接写入到 stdout
        Json::Value root = createSampleJson();
        std::cout << "=== Writing JSON to stdout ===\n";
        std::cout << writeToJsonString(root) << std::endl;

    } else if (operation == "writestr") {
        // 写入 JSON 到字符串
        Json::Value root = createSampleJson();
        std::string jsonStr = writeToJsonString(root);
        std::cout << "=== JSON as string ===\n" << jsonStr << std::endl;

    } else if (operation == "readstr") {
        // 测试从字符串解析 JSON
        std::string jsonStr = R"({"Name":"jack","Age":30,"Hobbies":["music","travel"],"Address":{"City":"Shanghai"}})";
        Json::Value root;
        if (parseJsonFromString(jsonStr, root)) {
            std::cout << "=== Parsed JSON from string ===\n";
            std::cout << "Name: " << root["Name"].asString() << "\n";
            std::cout << "Age: " << root["Age"].asInt() << "\n";
            std::cout << "First Hobby: " << root["Hobbies"][0].asString() << "\n";
            std::cout << "City: " << root["Address"]["City"].asString() << "\n";
        }

    } else if (operation == "writefile") {
        // 写入 JSON 到文件
        Json::Value root = createSampleJson();
        if (writeJsonToFile("example.json", root)) {
            std::cout << "JSON written to example.json\n";
        }

    } else if (operation == "readfile") {
        // 从文件中读取并解析 JSON
        Json::Value root;
        if (readJsonFromFile("example.json", root)) {
            std::cout << "=== Read JSON from file ===\n";
            std::cout << "Name: " << root["Name"].asString() << "\n";
            std::cout << "Age: " << root["Age"].asInt() << "\n";
            std::cout << "City: " << root["Address"]["City"].asString() << "\n";
        }

    } else {
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}