#pragma once

#define    RET_OK nullptr

#ifdef _WIN32
#include <Windows.h>
#include <direct.h>
#include <io.h>
#endif

#include <string>
#include <vector>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include "onnxruntime_cxx_api.h"
#include "utils.h"
#ifdef USE_CUDA
#include <cuda_fp16.h>
#endif





class SAM
{
public:
    SAM();

    ~SAM();

public:

    const char* CreateSession(SEG::DL_INIT_PARAM& iParams);

    const char* RunSession(cv::Mat& iImg, std::vector<SEG::DL_RESULT>& oResult, SEG::MODEL_TYPE modelType);

    char* WarmUpSession(SEG::MODEL_TYPE modelType);

    template<typename N>
    char* TensorProcess(clock_t& starttime_1, cv::Mat& iImg, N& blob, std::vector<int64_t>& inputNodeDims,
        SEG::MODEL_TYPE modelType, std::vector<SEG::DL_RESULT>& oResult, Utils& utilities);



    std::vector<std::string> classes{};

private:
    Ort::Env env;
    std::unique_ptr<Ort::Session> session;
    bool cudaEnable;
    Ort::RunOptions options;
    std::vector<const char*> inputNodeNames;
    std::vector<const char*> outputNodeNames;

    SEG::MODEL_TYPE modelType;
    std::vector<int> imgSize;
    float rectConfidenceThreshold;
    float iouThreshold;

};