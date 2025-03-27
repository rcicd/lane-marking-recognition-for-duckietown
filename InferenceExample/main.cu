#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <thread>
#include <stdexcept>
#include <cstring>
#include <sstream>
#include <csignal>
#include <atomic>

#include <opencv2/opencv.hpp>

// TensorRT
#include "NvInfer.h"
#include "NvInferRuntime.h"

// CUDA
#include <cuda_runtime_api.h>
#include <cuda.h>

// POSIX shared memory and semaphores
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <semaphore.h>
#include <unistd.h>

#define DEBUG_PERFOMANCE_BUFFER 30

using namespace nvinfer1;
using namespace std;

class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        }
    }
};


__global__ void preprocessKernel(const unsigned char* input, float* output,
                                 int inWidth, int inHeight, int channels,
                                 int cropY, int outWidth, int outHeight)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < outWidth && y < outHeight) {
        int origY = y + cropY;
        int inputIdx = (origY * inWidth + x) * channels;
        unsigned char B = input[inputIdx + 0];
        unsigned char G = input[inputIdx + 1];
        unsigned char R = input[inputIdx + 2];
        int outIdx = y * outWidth + x;
        output[0 * outWidth * outHeight + outIdx] = R / 255.0f;
        output[1 * outWidth * outHeight + outIdx] = G / 255.0f;
        output[2 * outWidth * outHeight + outIdx] = B / 255.0f;
    }
}


int main(int argc, char** argv)
{
    try {
        Logger trtLogger;
        std::string engineFile = "RoadMarkupsSegmentationNet.engine";
        std::ifstream engineStream(engineFile, std::ios::binary);
        if (!engineStream) {
            throw std::runtime_error("Failed to open engine file: " + engineFile);
        }
        engineStream.seekg(0, engineStream.end);
        size_t engineSize = engineStream.tellg();
        engineStream.seekg(0, engineStream.beg);
        std::vector<char> engineData(engineSize);
        engineStream.read(engineData.data(), engineSize);
        engineStream.close();

        IRuntime* runtime = createInferRuntime(trtLogger);
        if (!runtime)
            throw std::runtime_error("Failed to create TensorRT runtime");
        ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), engineSize, nullptr);
        if (!engine)
            throw std::runtime_error("Failed to deserialize CUDA engine");
        IExecutionContext* context = engine->createExecutionContext();
        if (!context)
            throw std::runtime_error("Failed to create execution context");

        std::cout << "TensorRT engine and context created" << std::endl;

        int inputIndex = -1;
        int outputIndex = -1;
        for (int i = 0; i < engine->getNbBindings(); ++i) {
            if (engine->bindingIsInput(i))
                inputIndex = i;
            else
                outputIndex = i;
        }
        if (inputIndex == -1 || outputIndex == -1)
            throw std::runtime_error("Engine does not have expected input/output");

        Dims inputDims = engine->getBindingDimensions(inputIndex);
        int batchSize = inputDims.d[0];
        int channels = inputDims.d[1];
        int inputH = inputDims.d[2];
        int inputW = inputDims.d[3];
        size_t inputSize = batchSize * channels * inputH * inputW * sizeof(float);

        Dims outputDims = engine->getBindingDimensions(outputIndex);
        size_t outputElemCount = 1;
        for (int i = 0; i < outputDims.nbDims; ++i)
            outputElemCount *= outputDims.d[i];
        size_t outputSize = outputElemCount * sizeof(int); // тип int32

        void* d_input = nullptr;
        void* d_output = nullptr;
        cudaMalloc(&d_input, inputSize);
        cudaMalloc(&d_output, outputSize);

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        std::vector<int> hostOutput(outputElemCount);

        const int cropY = 320;

        cv::Mat image = cv::imread(argv[1]);
        if (image.empty()) {
            std::cerr << "Failed to read image" << std::endl;
            return 1;
        }

        if (image.rows <= cropY) {
            std::cerr << "Image too small for cropping" << std::endl;
            return 1;
        }

        size_t size = image.total() * image.elemSize();
        unsigned char* d_cropped = nullptr;
        cudaMalloc(reinterpret_cast<void**>(&d_cropped), size);
        cudaMemcpyAsync(d_cropped, image.data, size, cudaMemcpyHostToDevice, stream);

        dim3 block(16, 16);
        dim3 grid((inputW + block.x - 1) / block.x, (inputH + block.y - 1) / block.y);
        preprocessKernel<<<grid, block, 0, stream>>>(d_cropped, reinterpret_cast<float*>(d_input),
                                                     inputW, image.rows, 3,
                                                     cropY, inputW, inputH);
        cudaFree(d_cropped);

        void* bindings[] = { d_input, d_output };
        context->enqueueV2(bindings, stream, nullptr);

        cudaMemcpyAsync(hostOutput.data(), d_output, outputSize, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        cv::Mat segMask(inputH, inputW, CV_32SC1, hostOutput.data());
        cv::Mat segMaskU8;
        segMask.convertTo(segMaskU8, CV_8U);
        cv::imwrite("out.png", segMaskU8);
        std::vector<uchar> pngBuffer;
        cv::imencode(".png", segMaskU8, pngBuffer);

        cudaStreamDestroy(stream);
        cudaFree(d_input);
        cudaFree(d_output);
        context->destroy();
        engine->destroy();
        runtime->destroy();


        return 0;
    }
    catch (std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return -1;
    }
}