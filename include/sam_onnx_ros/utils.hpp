#ifndef SAM_ONNX_ROS_UTILS_HPP_
#define SAM_ONNX_ROS_UTILS_HPP_

#define RET_OK nullptr

#include "onnxruntime_cxx_api.h"
#include <sam_onnx_ros/config.hpp>
#include <sam_onnx_ros/dl_types.hpp>

#if defined(SAM_ONNX_ROS_CUDA_ENABLED) && SAM_ONNX_ROS_CUDA_ENABLED
#include <cuda_fp16.h>
#endif

#include <cstdio>
#include <vector>

class Utils
{
public:
    Utils();
    ~Utils();

    char* PreProcess(const cv::Mat& iImg, std::vector<int> iImgSize, cv::Mat& oImg);
    void ScaleBboxPoints(const cv::Mat& iImg, std::vector<int> iImgSize, std::vector<float>& pointCoords, std::vector<float>& PointsCoordsScaled);

    std::vector<Ort::Value> PrepareInputTensor(Ort::Value& decoderInputTensor, std::vector<float>& pointCoordsScaled, std::vector<int64_t> pointCoordsDims,
                                               std::vector<float>& pointLabels, std::vector<int64_t> pointLabelsDims, std::vector<float>& maskInput,
                                               std::vector<int64_t> maskInputDims, std::vector<float>& hasMaskInput, std::vector<int64_t> hasMaskInputDims);

    void PostProcess(std::vector<Ort::Value>& output_tensors, const cv::Mat& iImg, std::vector<int> iImgSize, SEG::DL_RESULT& result);

    // Definition: Flattened image to blob (and normalizaed) for deep learning inference. Also reorganize from HWC to CHW.
    // Note: Code in header file since it is used outside of this utils src code.
    template <typename T>
    char* BlobFromImage(const cv::Mat& iImg, T& iBlob)
    {
        int channels = iImg.channels();
        int imgHeight = iImg.rows;
        int imgWidth = iImg.cols;

        for (int c = 0; c < channels; c++)
        {
            for (int h = 0; h < imgHeight; h++)
            {
                for (int w = 0; w < imgWidth; w++)
                {
                    iBlob[c * imgWidth * imgHeight + h * imgWidth + w] = typename std::remove_pointer<T>::type(
                        (iImg.at<cv::Vec3b>(h, w)[c]) / 255.0f);
                }
            }
        }
        return RET_OK;
    }

private:
    float resizeScales_;
    float resizeScalesBbox_; // letterbox scale
};

#endif // SAM_ONNX_ROS_UTILS_HPP_
