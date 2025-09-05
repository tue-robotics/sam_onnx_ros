#ifndef UTILS_H
#define UTILS_H

#define RET_OK nullptr

#include <string>
#include <vector>
#include <cstdio>
#include "onnxruntime_cxx_api.h"
#include "dl_types.h"
#ifdef USE_CUDA
#include <cuda_fp16.h>
#endif

class Utils
{
public:
    Utils();
    ~Utils();

    char *PreProcess(const cv::Mat &iImg, std::vector<int> iImgSize, cv::Mat &oImg);
    void ScaleBboxPoints(const cv::Mat &iImg, std::vector<int> iImgSize, std::vector<float> &pointCoords, std::vector<float> &PointsCoordsScaled);

    std::vector<Ort::Value> PrepareInputTensor(Ort::Value &decoderInputTensor, std::vector<float> &pointCoordsScaled, std::vector<int64_t> pointCoordsDims,
                                               std::vector<float> &pointLabels, std::vector<int64_t> pointLabelsDims, std::vector<float> &maskInput,
                                               std::vector<int64_t> maskInputDims, std::vector<float> &hasMaskInput, std::vector<int64_t> hasMaskInputDims);

    void PostProcess(std::vector<Ort::Value> &output_tensors, const cv::Mat &iImg, std::vector<int> iImgSize, SEG::DL_RESULT &result);

    // Definition: Flattened image to blob (and normalizaed) for deep learning inference. Also reorganize from HWC to CHW.
    // Note: Code in header file since it is used outside of this utils src code.
    template <typename T>
    char *BlobFromImage(const cv::Mat &iImg, T &iBlob)
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
    float _resizeScales;
    float _resizeScalesBbox; // letterbox scale
};

#endif // UTILS_H