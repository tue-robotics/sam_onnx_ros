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
    std::cout << params << "params" <<std::endl;
    samSegmentorEncoder->CreateSession(params);


    DL_INIT_PARAM params1;
    params1 = params;
    params1.imgSize = { 256, 64, 64 };
    params1.modelPath = "/home/amigo/Documents/repos/hero_sam/sam_inference/model/SAM_mask_decoder.onnx";
    std::cout << params1 << "params1" << std::endl;
    //params.modelPath = "/home/amigo/Documents/repos/hero_sam/sam_inference/model/FastSAM-x.onnx";
    samSegmentorDecoder->CreateSession(params1);


    //Running inference
    std::filesystem::path current_path = std::filesystem::current_path();
    std::filesystem::path imgs_path = current_path / "sam_inference/build/images";
    for (auto& i : std::filesystem::directory_iterator(imgs_path))
    {
        if (i.path().extension() == ".jpg" || i.path().extension() == ".png" || i.path().extension() == ".jpeg")
        {
            std::string img_path = i.path().string();
            cv::Mat img = cv::imread(img_path);
            std::vector<DL_RESULT> res;
            samSegmentorEncoder->RunSession(img, res);

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
    //ClsTest();
}