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
    std::tie(samSegmentors, params_encoder, params_decoder) = Initializer();
    std::filesystem::path current_path = std::filesystem::current_path();
    std::filesystem::path imgs_path = current_path / "../../hero_sam/pipeline/build/images";
    for (auto &i : std::filesystem::directory_iterator(imgs_path))
    {
        if (i.path().extension() == ".jpg" || i.path().extension() == ".png" || i.path().extension() == ".jpeg")
        {
            std::string img_path = i.path().string();
            cv::Mat img = cv::imread(img_path);
            std::vector<cv::Mat> masks;
            masks = SegmentAnything(samSegmentors, params_encoder, params_decoder, img);

        }
    }
    return 0;
}