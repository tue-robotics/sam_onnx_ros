#include "segmentation.h"

std::tuple<std::vector<std::unique_ptr<SAM>>, SEG::DL_INIT_PARAM, SEG::DL_INIT_PARAM> Initializer()
{
    std::vector<std::unique_ptr<SAM>> samSegmentors;
    samSegmentors.push_back(std::make_unique<SAM>());
    samSegmentors.push_back(std::make_unique<SAM>());

    std::unique_ptr<SAM> samSegmentorEncoder = std::make_unique<SAM>();
    std::unique_ptr<SAM> samSegmentorDecoder = std::make_unique<SAM>();
    SEG::DL_INIT_PARAM params_encoder;
    SEG::DL_INIT_PARAM params_decoder;

    params_encoder.rectConfidenceThreshold = 0.1;
    params_encoder.iouThreshold = 0.5;
    params_encoder.modelPath = "SAM_encoder.onnx";
    params_encoder.imgSize = { 1024, 1024 };

    params_decoder = params_encoder;
    params_decoder.modelType = SEG::SAM_SEGMENT_DECODER;
    params_decoder.modelPath = "SAM_mask_decoder.onnx";



    #ifdef USE_CUDA
    params_encoder.cudaEnable = true;
    #else
    params_encoder.cudaEnable = false;
    #endif

    samSegmentorEncoder->CreateSession(params_encoder);
    samSegmentorDecoder->CreateSession(params_decoder);
    samSegmentors[0] = std::move(samSegmentorEncoder);
    samSegmentors[1] = std::move(samSegmentorDecoder);
    return {std::move(samSegmentors), params_encoder, params_decoder};
}

void SegmentAnything(std::vector<std::unique_ptr<SAM>>& samSegmentors, SEG::DL_INIT_PARAM& params_encoder, SEG::DL_INIT_PARAM& params_decoder, cv::Mat& img) {

    std::vector<SEG::DL_RESULT> resSam;
    SEG::DL_RESULT res;

    SEG::MODEL_TYPE modelTypeRef = params_encoder.modelType;
    samSegmentors[0]->RunSession(img, resSam, modelTypeRef, res);


    modelTypeRef = params_decoder.modelType;
    samSegmentors[1]->RunSession(img, resSam, modelTypeRef, res);
    std::cout << "Press any key to exit" << std::endl;
    cv::imshow("Result of Detection", img);
    cv::waitKey(0);
    cv::destroyAllWindows();
}
