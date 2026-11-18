//trtexec --onnx=movinet_a0_dumping.onnx --saveEngine=movinet_a0_dumping.engine

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <string>
#include <vector>
#include <algorithm>

const std::string modelPath = "src/movinet_a0_dumping.engine";
const std::string videoPath = "DATASET/videos/vid0005.mp4";
const std::string outputPath = "OUTPUT";

const float THRESHOLD = 0.5;
const int WIDTH = 172;
const int HEIGHT = 172;
const int WINDOW_SIZE = 50; // 50 frame mot khung
const int WINDOW_STEP = 25; // 25 frame mot buoc nhay
const double SAMPLE_FPS = 5.0; // Lay 5 frame mot giay
const double WINDOW_SEC = 10.0; // 10 giay mot cua so
const size_t NUM_INPUT = 3 * WINDOW_SIZE * WIDTH * HEIGHT; //kich thuoc dau vao: 3 kenh mau * 50 frame * 172 * 172

// Khoi tao logger, chi in ra canh bao
struct Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << message << std::endl;
        }
    }
};
Logger logger;

// Thoi gian vut rac (giay bat dau, giay ket thuc)
struct DumpingTime{
    double start;
    double end;
};

// Ham kiem tra loi CUDA
bool checkCudaErrorCode(cudaError_t error, std::string where){
    if(error != cudaSuccess){
        std::cerr << "Loi CUDA: " << cudaGetErrorString(error) << " tai " << where << std::endl;
        return false;
    }
    return true;
}

// Ham tai model vao vung nho dem
std::vector<char> loadModel(const std::string modelPath){
    std::ifstream file(modelPath, std::ios::binary); // binary do khong phai la file text
    if (!file) {
        return {};
    }

    file.seekg(0, std::ios::end);
    size_t modelSize = file.tellg();
    file.seekg(0);

    // Cap phat bo nho dem cho mo hinh
    std::vector<char> buffer(modelSize);
    file.read(buffer.data(), modelSize); // Ghi engine vao vung nho dem
    return buffer;
}

// Ham in ten, vi tri va kich thuoc cua cac tensor dau vao va dau ra
void getTensors(nvinfer1::ICudaEngine* engine) {
    for (int i = 0; i < engine->getNbIOTensors(); i++) { // Lap qua cac tensor
        const char* name = engine->getIOTensorName(i);
        bool isInput = engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
        bool isOutput = engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT;

        nvinfer1::Dims shape = engine->getTensorShape(name);

        if(isInput) std::cout << "Tensor dau vao, kich thuoc: ";
        if(isOutput) std::cout << "Tensor dau ra, kich thuoc: ";

        for (int k = 0; k < shape.nbDims; k++) {    // Lap qua cac chieu cua 1 tensor
            std::cout << shape.d[k];
            if (k < shape.nbDims - 1) std::cout << " x ";
        }
        std::cout << std::endl;
    }
}

// Ham tien xu ly cac frame
cv::Mat preprocess(cv::Mat frame){
    cv::Mat resized, colorized, normalize;
    cv::resize(frame, resized, cv::Size(WIDTH, HEIGHT));
    cv::cvtColor(resized, colorized, cv::COLOR_BGR2RGB);
    colorized.convertTo(normalize, CV_32FC3, 1.0 / 255.0);
    return normalize;
}

// Ham lay mau cac frame tu video
std::vector<cv::Mat> readFrames(const std::string& videoPath) {
    std::vector<cv::Mat> frames;

    cv::VideoCapture cap(videoPath);
    if(cap.isOpened() == false) {
        std::cerr << "Khong mo duoc video: " << videoPath << std::endl;
        return {};
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    if(fps < 0) fps = 25.0;
    double step = fps / SAMPLE_FPS;
    double nextFrame = 0;
    int frameIndex = 0;

    cv::Mat frame;
    while(cap.read(frame)){
        if(frameIndex >= nextFrame){
            frames.push_back(preprocess(frame));
            nextFrame+=step;
        }
        frameIndex++;
    }
    return frames;
}

// Ham tim vi tri bat dau cua moi khung
std::vector<int> windowStartFrame(int numFrame){
    std::vector<int> startFrames;
    for(int i = 0; i + WINDOW_SIZE <= numFrame; i+=WINDOW_STEP){
        startFrames.push_back(i);
    }
    if(startFrames.empty() || startFrames.back() + WINDOW_SIZE < numFrame){ // empty la truong hop video < 10s
        startFrames.push_back(std::max(0, numFrame - WINDOW_SIZE));
    }
    return startFrames;
}

// Xep 50 khung vao buffer theo thu tu: kenh, khung, hang, cot (NCTHW)
void packClip(const std::vector<cv::Mat>& frames, int startFrame, std::vector<float>& buffer) {
    std::vector<cv::Mat> clip;
    int frameSize = (int)frames.size();
    for(int i = 0; i < WINDOW_SIZE; i++){
        clip.push_back(frames[std::min(startFrame + i, frameSize - 1)]); // Lap lai frame cuoi neu khong du frame
    }

    // Sap xep lai cac phan tu theo trat tu kenh, khung, hang, cot
    cv::Mat blob = cv::dnn::blobFromImages(clip); // NCHW: khung, kenh, hang, cot
    cv::Mat clipTensor;
    cv::transposeND(blob, {1, 0, 2, 3}, clipTensor); // kenh, khung, hang, cot

    float* data = clipTensor.ptr<float>();
    std::copy(data, data + NUM_INPUT, buffer.begin());
}

// Ham chay inference tren 1 khung, tra ve xac suat co vut rac
float runWindowInference(nvinfer1::IExecutionContext* context, void* gpuInput, void* gpuOutput, cudaStream_t stream, const std::vector<float>& buffer){
    float possibility = -1.0;
    cudaError_t err1 = cudaMemcpyAsync(gpuInput, buffer.data(), NUM_INPUT * (sizeof(float)), cudaMemcpyHostToDevice, stream); // Ram to GPU

    bool enqueue = context -> enqueueV3(stream); // Dua lenh chay mo hinh vao GPU
    if(enqueue == false) return -1.0;

    cudaError_t err2 = cudaMemcpyAsync(&possibility, gpuOutput, sizeof(float), cudaMemcpyDeviceToHost, stream); // GPU ve Ram
    cudaError_t err3 = cudaStreamSynchronize(stream);

    if(!checkCudaErrorCode(err1, "chep du lieu dau vao") || !checkCudaErrorCode(err2, "chep du lieu dau ra") || !checkCudaErrorCode(err3, "chep lenh chay engine len GPU")){
        return -1.0;
    }
    return possibility;
}

// Ham chay inference tren toan bo video: chay moi khung, gop cac khung co do rac thanh khoang thoi gian
bool runInference(const std::string& videoPath, nvinfer1::IExecutionContext* context, void* gpuInput, void* gpuOutput, cudaStream_t stream, std::vector<DumpingTime>& dumpingTimes){
    std::vector<cv::Mat> frames = readFrames(videoPath);
    if(frames.empty()){
        std::cerr << "Khong doc duoc video: " << videoPath << std::endl;
        return false;
    }
    int numFrames = (int)frames.size();
    std::cout << "\nVideo: " << videoPath << " - " << (numFrames / SAMPLE_FPS) << " giay" << std::endl;

    std::vector<float> buffer(NUM_INPUT); // Bo nho chua dau vao
    std::vector<DumpingTime> time; // Chua cac khoang thoi gian co do rac

    std::vector<int> startFrames = windowStartFrame(numFrames); // Lay vi tri bat dau cua moi khung
    for(int i = 0; i < (int)startFrames.size(); i++){
        packClip(frames, startFrames[i], buffer);

        float possibility = runWindowInference(context, gpuInput, gpuOutput, stream, buffer);
        if(possibility < 0){
            std::cerr << "Khong chay duoc Inference cho 1 khung do gap loi CUDA" << std::endl;
            return false;
        }

        double startSecond = startFrames[i] / SAMPLE_FPS;
        double endSecond = std::min(startFrames[i] + WINDOW_SIZE, numFrames) / SAMPLE_FPS;
        std::cout << "Cua so: " << startSecond << "s den " << endSecond << "s co xac suat " << possibility << std::endl;

        if(possibility >= THRESHOLD){
            if(time.empty() == false && time.back().end >= startSecond){
                time.back().end = std::max(endSecond, time.back().end);
            }else{
                time.push_back({startSecond, endSecond});
            }
        }
    }
    if(time.empty() == true){
        std::cout << "\nKet qua: Khong phat hien ra do rac trong video" << std::endl;
    }else{
        std::cout << "\nKet qua: Phat hien do rac tai cac khoang thoi gian:" << std::endl;
        for(int i = 0; i < (int)time.size(); i++){
            std::cout << "- " << time[i].start << "s den " << time[i].end << "s" << std::endl;
        }
    }
    return true;
}

int main(){
    // Tai model
    std::vector<char> model = loadModel(modelPath);
    if(model.empty()){
        std::cerr << "Khong tai duoc model:" << modelPath << std::endl;
        return -1;
    }

    // Tao Runtime de doc file engine
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    if(!runtime){
        std::cerr << "Khong tao duoc runtime" << std::endl;
        return -1;
    }

    // Doc engine tu vung nho dem
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(model.data(), model.size());
    if(!engine){
        std::cerr << "Khong tao duoc engine" << std::endl;
        delete(runtime);
        return -1;
    }

    // Trat tu sap xep dau vao suy ra tu day
    getTensors(engine); // In ra ten va kich thuoc cua cac tensor dau vao va dau ra

    // Tao phien lam viec de chay engine
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    if(!context){
        std::cerr << "Khong tao duoc context" << std::endl;
        delete(engine);
        delete(runtime);
        return -1;
    }

    // Cap phat bo nho GPU cho input, output va stream
    void* gpuInput = nullptr;
    void* gpuOutput = nullptr;
    cudaStream_t stream = nullptr;
    cudaError_t err = cudaMalloc(&gpuInput, NUM_INPUT * sizeof(float));
    cudaError_t err2 = cudaMalloc(&gpuOutput, sizeof(float));
    cudaError_t err3 = cudaStreamCreate(&stream);
    if(!checkCudaErrorCode(err, "Khoi tao bo nho dau vao") || !checkCudaErrorCode(err2, "Khoi tao bo nho dau ra") || !checkCudaErrorCode(err3, "Khoi tao bo nho stream")){
        delete(context);
        delete(engine);
        delete(runtime);
        return -1;
    }

    // Gan dia chi bo nho GPU vao tensor
    if(!context->setTensorAddress("video", gpuInput) || !context->setTensorAddress("prob", gpuOutput)){
        std::cerr << "Khong gan duoc dia chi bo nho GPU cho tensor" << std::endl;
        cudaFree(gpuInput);
        cudaFree(gpuOutput);
        cudaStreamDestroy(stream);
        delete(context);
        delete(engine);
        delete(runtime);
        return -1;
    }

    std::vector<DumpingTime> time;

    bool runSuccess = runInference(videoPath, context, gpuInput, gpuOutput, stream, time);

    if(stream) cudaStreamDestroy(stream);
    if(gpuInput) cudaFree(gpuInput);
    if(gpuOutput) cudaFree(gpuOutput);
    delete context;
    delete engine;
    delete runtime;

    return runSuccess ? 0 : -1;
}