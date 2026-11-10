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
    ~Utils() = default;

    char* preProcess(const cv::Mat& iImg, std::vector<int> iImgSize, cv::Mat& oImg);
    void scaleBboxPoints(const cv::Mat& iImg,
                         std::vector<int> iImgSize,
                         std::vector<float>& pointCoords,
                         std::vector<float>& pointCoordsScaled);

    static std::vector<Ort::Value> prepareInputTensor(Ort::Value& decoderInputTensor,
                                                      std::vector<float>& pointCoordsScaled,
                                                      std::vector<int64_t> pointCoordsDims,
                                                      std::vector<float>& pointLabels,
                                                      std::vector<int64_t> pointLabelsDims,
                                                      std::vector<float>& maskInput,
                                                      std::vector<int64_t> maskInputDims,
                                                      std::vector<float>& hasMaskInput,
                                                      std::vector<int64_t> hasMaskInputDims);

    static void postProcess(std::vector<Ort::Value>& output_tensors,
                            const cv::Mat& iImg,
                            std::vector<int> iImgSize,
                            SEG::DlResult& result);

    // Definition: Flattened image to blob (and normalizaed) for deep learning inference. Also reorganize from HWC to
    // CHW. Note: Code in header file since it is used outside of this utils src code.
    template <typename T> char* blobFromImage(const cv::Mat& iImg, T& iBlob)
    {
        int const channels = iImg.channels();
        int const img_height = iImg.rows;
        int const img_width = iImg.cols;

        for (int c = 0; c < channels; c++)
        {
            for (int h = 0; h < img_height; h++)
            {
                for (int w = 0; w < img_width; w++)
                {
                    iBlob[(c * img_width * img_height) + (h * img_width) + w] =
                        std::remove_pointer_t<T>(static_cast<float>(iImg.at<cv::Vec3b>(h, w)[c]) / 255.0F);
                }
            }
        }
        return RET_OK;
    }

private:
    float resizeScales_{1.0F};
    float resizeScalesBbox_{1.0F}; // letterbox scale
};

#endif // SAM_ONNX_ROS_UTILS_HPP_
