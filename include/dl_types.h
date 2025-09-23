#ifndef DL_TYPES_H
#define DL_TYPES_H

#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core/types.hpp>
#include <vector>
#include <string>

namespace SEG
{
    enum MODEL_TYPE
    {
        SAM_SEGMENT_ENCODER = 1,
        SAM_SEGMENT_DECODER = 2,
    };

    typedef struct _DL_INIT_PARAM
    {
        // Yolo & Common Part
        std::string modelPath;
        MODEL_TYPE modelType = SAM_SEGMENT_ENCODER;
        std::vector<int> imgSize = {640, 640};
        bool cudaEnable = false;
        int logSeverityLevel = 3;
        int intraOpNumThreads = 1;

        // Overloaded output operator for _DL_INIT_PARAM to print its contents
        friend std::ostream &operator<<(std::ostream &os, const _DL_INIT_PARAM &param)
        {
            os << "modelPath: " << param.modelPath << "\n";
            os << "modelType: " << param.modelType << "\n";
            os << "imgSize: ";
            for (const auto &size : param.imgSize)
                os << size << " ";
            os << "\n";
            os << "cudaEnable: " << (param.cudaEnable ? "true" : "false") << "\n";
            os << "logSeverityLevel: " << param.logSeverityLevel << "\n";
            os << "intraOpNumThreads: " << param.intraOpNumThreads;
            return os;
        }

    } DL_INIT_PARAM;

    typedef struct _DL_RESULT
    {
        // For SAM encoder model, this will be filled with detected boxes from object detection model.
        std::vector<cv::Rect> boxes;
        std::vector<float> embeddings;
        std::vector<cv::Mat> masks;

    } DL_RESULT;
} // namespace SEG
#endif // DL_TYPES_H
