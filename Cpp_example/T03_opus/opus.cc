#include <iostream>
#include <fstream>
#include <string>
extern "C" {
#include <opus.h>
}

// 配置参数
#define FRAME_SIZE 960       // 20ms frame at 48kHz
#define SAMPLE_RATE 48000    // Sample rate (Hz)
#define CHANNELS 1           // Mono
#define APPLICATION OPUS_APPLICATION_AUDIO
#define BITRATE 16000        // Bitrate in bps (语音推荐值)

class OpusTest {
public:
    OpusTest() : encoder(nullptr), decoder(nullptr) {}
    ~OpusTest();

    bool init();
    void run(const std::string& inputPath, const std::string& pcmOutputPath);

private:
    OpusEncoder* encoder;
    OpusDecoder* decoder;

    void handleErrors(const std::string& message, int errorCode = 0);
};

bool OpusTest::init() {
    int error;

    // 创建编码器
    encoder = opus_encoder_create(SAMPLE_RATE, CHANNELS, APPLICATION, &error);
    if (error != OPUS_OK || !encoder) {
        handleErrors("Failed to create encoder", error);
        return false;
    }

    // 设置比特率
    opus_encoder_ctl(encoder, OPUS_SET_BITRATE(BITRATE));

    // 创建解码器
    decoder = opus_decoder_create(SAMPLE_RATE, CHANNELS, &error);
    if (error != OPUS_OK || !decoder) {
        handleErrors("Failed to create decoder", error);
        return false;
    }

    return true;
}

void OpusTest::handleErrors(const std::string& message, int errorCode) {
    std::cerr << message << std::endl;
    if (errorCode != 0) {
        std::cerr << "Error code: " << opus_strerror(errorCode) << std::endl;
    }
}

void OpusTest::run(const std::string& inputPath, const std::string& pcmOutputPath) {
    std::ifstream fin(inputPath, std::ios::binary);
    std::ofstream fout(pcmOutputPath, std::ios::binary);
    std::ofstream opusOut("output.opus", std::ios::binary);

    if (!fin.is_open() || !fout.is_open() || !opusOut.is_open()) {
        std::cerr << "Failed to open input/output file." << std::endl;
        return;
    }

    opus_int16 pcmBuffer[FRAME_SIZE];
    unsigned char encoded[1024];     // 编码缓冲区
    opus_int16 decoded[FRAME_SIZE];  // 解码缓冲区

    while (true) {
        fin.read(reinterpret_cast<char*>(pcmBuffer), sizeof(pcmBuffer));
        std::streamsize readBytes = fin.gcount();

        if (readBytes <= 0)
            break;

        int samples = readBytes / sizeof(opus_int16);

        // 编码
        int encodedSize = opus_encode(encoder, pcmBuffer, samples, encoded, sizeof(encoded));
        if (encodedSize < 0) {
            handleErrors("Encoding failed", encodedSize);
            continue;
        }

        // 写入 Opus 文件
        opusOut.write(reinterpret_cast<char*>(encoded), encodedSize);

        // 解码
        int decodedSamples = opus_decode(decoder, encoded, encodedSize, decoded, FRAME_SIZE, 0);
        if (decodedSamples < 0) {
            handleErrors("Decoding failed", decodedSamples);
            continue;
        }

        // 写入输出文件
        fout.write(reinterpret_cast<char*>(decoded), decodedSamples * sizeof(opus_int16));
    }

    fin.close();
    fout.close();
    opusOut.close();

    // 统计压缩率
    std::ifstream finStat(inputPath, std::ios::binary);
    finStat.seekg(0, std::ios::end);
    size_t pcmSize = finStat.tellg();

    std::ifstream opusStat("output.opus", std::ios::binary);
    opusStat.seekg(0, std::ios::end);
    size_t opusSize = opusStat.tellg();

    double compressionRatio = (double)pcmSize / opusSize;

    std::cout << "PCM file size: " << pcmSize << " bytes" << std::endl;
    std::cout << "Opus file size: " << opusSize << " bytes" << std::endl;
    std::cout << "Compression ratio: " << compressionRatio << "x" << std::endl;
}

OpusTest::~OpusTest() {
    if (encoder) opus_encoder_destroy(encoder);
    if (decoder) opus_decoder_destroy(decoder);
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input.pcm> <output.pcm>" << std::endl;
        return -1;
    }

    OpusTest test;
    if (!test.init()) {
        std::cerr << "Initialization failed." << std::endl;
        return -1;
    }

    test.run(argv[1], argv[2]);

    std::cout << "Opus encode/decode test completed successfully." << std::endl;
    return 0;
}