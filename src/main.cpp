#include <iostream>
#include <iomanip>
#include "inference.h"
#include <filesystem>
#include <fstream>
#include <random>



void SegmentAnything() {

    SAM* samSegmentorEncoder = new SAM;
    SAM* samSegmentorDecoder = new SAM;
    DL_INIT_PARAM params;
    params.rectConfidenceThreshold = 0.1;
    params.iouThreshold = 0.5;
    params.modelPath = "/home/amigo/Documents/repos/hero_sam/sam_inference/model/SAM_encoder.onnx";
    params.imgSize = { 1024, 1024 };

    samSegmentorEncoder->CreateSession(params);

    DL_INIT_PARAM params1;
    params1 = params;
    params.modelPath = "/home/amigo/Documents/repos/hero_sam/sam_inference/model/SAM_mask_decoder.onnx";
    samSegmentorDecoder->CreateSession(params1);

    //Running inference
    std::filesystem::path current_path = std::filesystem::current_path();
    std::filesystem::path imgs_path = current_path / "images";
    for (auto& i : std::filesystem::directory_iterator(imgs_path))
    {
        if (i.path().extension() == ".jpg" || i.path().extension() == ".png" || i.path().extension() == ".jpeg")
        {
            std::string img_path = i.path().string();
            cv::Mat img = cv::imread(img_path);
            std::vector<DL_RESULT> res;
            samSegmentorEncoder->RunSession(img, res);
        }
    }
    //input_tensor_size = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementCount();
    //output_tensor_size = session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementCount();
    //input_shape = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    //output_shape = session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
}

int main()
{
    SegmentAnything();
    //ClsTest();
}