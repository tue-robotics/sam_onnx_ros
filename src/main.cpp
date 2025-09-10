#include "segmentation.h"
#include <iostream>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>

int main()
{
    // Running inference
    std::vector<std::unique_ptr<SAM>> samSegmentors;
    SEG::DL_INIT_PARAM params_encoder;
    SEG::DL_INIT_PARAM params_decoder;
    std::vector<SEG::DL_RESULT> resSam;
    SEG::DL_RESULT res;
    std::tie(samSegmentors, params_encoder, params_decoder, res, resSam) = Initializer();
    std::filesystem::path current_path = std::filesystem::current_path();
    std::filesystem::path imgs_path =  "/home/amigo/Documents/repos/hero_sam/sam_inference/build/images"; // current_path / <- you could use
    for (auto &i : std::filesystem::directory_iterator(imgs_path))
    {
        if (i.path().extension() == ".jpg" || i.path().extension() == ".png" || i.path().extension() == ".jpeg")
        {
            std::string img_path = i.path().string();
            cv::Mat img = cv::imread(img_path);

            SegmentAnything(samSegmentors, params_encoder, params_decoder, img, resSam, res);

        }
    }
    return 0;
}