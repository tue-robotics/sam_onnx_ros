#ifndef SAM_ONNX_ROS_SAM_INFERENCE_HPP_
#define SAM_ONNX_ROS_SAM_INFERENCE_HPP_

#define RET_OK nullptr

#include "sam_onnx_ros/config.hpp"
#include "sam_onnx_ros/utils.hpp"

#if defined(SAM_ONNX_ROS_CUDA_ENABLED) && SAM_ONNX_ROS_CUDA_ENABLED
#include <cuda_fp16.h>
#endif

#include <cstdio>
#include <memory>
#include <vector>

class SAM
{
public:
    SAM();

    ~SAM();

public:
    const char* CreateSession(SEG::DL_INIT_PARAM& iParams);

    const char* RunSession(const cv::Mat& iImg, std::vector<SEG::DL_RESULT>& oResult, SEG::MODEL_TYPE modelType, SEG::DL_RESULT& result);

private:

    char* WarmUpSession_(SEG::MODEL_TYPE modelType);

    template <typename N>
    const char* TensorProcess_(clock_t& starttime_1, const cv::Mat& iImg, N& blob, std::vector<int64_t>& inputNodeDims,
                        SEG::MODEL_TYPE modelType, std::vector<SEG::DL_RESULT>& oResult, Utils& utilities, SEG::DL_RESULT& result);

    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    bool cudaEnable_;
    Ort::RunOptions options_;
    std::vector<const char*> inputNodeNames_;
    std::vector<const char*> outputNodeNames_;

    SEG::MODEL_TYPE modelType_;
    std::vector<int> imgSize_;
    float rectConfidenceThreshold_;
};

#endif // SAM_ONNX_ROS_SAM_INFERENCE_HPP_
