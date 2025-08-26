#pragma once
namespace SEG
{
    enum MODEL_TYPE
    {
        // FLOAT32 MODEL
        SAM_SEGMENT_ENCODER = 1,
        SAM_SEGMENT_DECODER = 2,
        // YOLO_CLS = 3,

        // FLOAT16 MODEL
        // YOLO_DETECT_V8_HALF = 4,
        // YOLO_POSE_V8_HALF = 5,
        // YOLO_CLS_HALF = 6
    };

    typedef struct _DL_INIT_PARAM
    {
        // Yolo & Common Part
        std::string modelPath;
        MODEL_TYPE modelType = SAM_SEGMENT_ENCODER;
        std::vector<int> imgSize = {640, 640};
        float rectConfidenceThreshold = 0.6;
        float iouThreshold = 0.5;
        int keyPointsNum = 2; // Note:kpt number for pose
        bool cudaEnable = false;
        int logSeverityLevel = 3;
        int intraOpNumThreads = 1;
        // std::vector<cv::Rect> boxes; // For SAM encoder model, this will be filled with detected boxes

        // Overloaded output operator for _DL_INIT_PARAM to print its contents
        friend std::ostream &operator<<(std::ostream &os, _DL_INIT_PARAM &param)
        {
            os << "modelPath: " << param.modelPath << "\n";
            os << "modelType: " << param.modelType << "\n";
            os << "imgSize: ";
            for (const auto &size : param.imgSize)
                os << size << " ";
            os << "\n";
            os << "rectConfidenceThreshold: " << param.rectConfidenceThreshold << "\n";
            os << "iouThreshold: " << param.iouThreshold << "\n";
            os << "keyPointsNum: " << param.keyPointsNum << "\n";
            os << "cudaEnable: " << (param.cudaEnable ? "true" : "false") << "\n";
            os << "logSeverityLevel: " << param.logSeverityLevel << "\n";
            os << "intraOpNumThreads: " << param.intraOpNumThreads;
            return os;
        }

    } DL_INIT_PARAM;

    typedef struct _DL_RESULT
    {

        // Yolo Part
        int classId;
        float confidence;
        std::vector<cv::Rect> boxes; // For SAM encoder model, this will be filled with detected boxes
        std::vector<cv::Point2f> keyPoints;

        // Sam Part
        std::vector<float> embeddings;
        // Masks for SAM decoder model output
        std::vector<cv::Mat> masks; // Each cv::Mat represents a mask

    } DL_RESULT;
} // namespace SEG