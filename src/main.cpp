#include <iostream>
#include <iomanip>
#include "sam_inference.h"
#include <filesystem>
#include <fstream>
#include <random>



void SegmentAnything() {

    SAM* samSegmentor = new SAM;
    SEG::DL_INIT_PARAM params;
    SEG::DL_INIT_PARAM params1;

    params.rectConfidenceThreshold = 0.1;
    params.iouThreshold = 0.5;
    params.modelPath = "/home/amigo/Documents/repos/hero_sam/sam_inference/model/SAM_encoder.onnx";
    params.imgSize = { 1024, 1024 };

    params1 = params;
    params1.modelType = SEG::SAM_SEGMENT_DECODER;
    params1.modelPath = "/home/amigo/Documents/repos/hero_sam/sam_inference/model/SAM_mask_decoder.onnx";


    #ifdef USE_CUDA
    params.cudaEnable = true;
    #else
    params.cudaEnable = false;
    #endif



    //Running inference
    std::filesystem::path current_path = std::filesystem::current_path();
    std::filesystem::path imgs_path = current_path / "images";
    for (auto& i : std::filesystem::directory_iterator(imgs_path))
    {
        if (i.path().extension() == ".jpg" || i.path().extension() == ".png" || i.path().extension() == ".jpeg")
        {
            std::string img_path = i.path().string();
            cv::Mat img = cv::imread(img_path);
            std::vector<SEG::DL_RESULT> res;
            samSegmentor->CreateSession(params);
            SEG::MODEL_TYPE modelTypeRef = params.modelType;
            samSegmentor->RunSession(img, res, modelTypeRef);




            samSegmentor->CreateSession(params1);
            modelTypeRef = params1.modelType;
            samSegmentor->RunSession(img, res, modelTypeRef);
            std::cout << "Press any key to exit" << std::endl;
            cv::imshow("Result of Detection", img);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
    }
}

int main()
{
    SegmentAnything();
}