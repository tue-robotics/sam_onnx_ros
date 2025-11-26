#include "sam_onnx_ros/segmentation.hpp"

#include <opencv2/opencv.hpp>

#include <filesystem>

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cerr << "Not enough args provided" << std::endl;
        return 1;
    }

    // Running inference
    std::vector<std::unique_ptr<SAM>> samSegmentors;
    SEG::DL_INIT_PARAM params_encoder;
    SEG::DL_INIT_PARAM params_decoder;
    std::vector<SEG::DL_RESULT> resSam;
    SEG::DL_RESULT res;

    const std::filesystem::path encoder_name = argv[1];
    const std::filesystem::path decoder_name = argv[2];

    std::tie(samSegmentors, params_encoder, params_decoder, res, resSam) = Initialize(encoder_name, decoder_name);

    std::filesystem::path imgs_path = argv[3];
    for (auto& i : std::filesystem::directory_iterator(imgs_path))
    {
        if (i.path().extension() == ".jpg" || i.path().extension() == ".png" || i.path().extension() == ".jpeg")
        {
            std::string img_path = i.path().string();
            cv::Mat img = cv::imread(img_path);

            SegmentAnything(samSegmentors, params_encoder, params_decoder, img, resSam, res);
            #ifdef LOGGING
            for (const auto& result : results)
            {
                std::cout << "Image path:   " << img_path << "\n"
                          << "# boxes:      " << result.boxes.size() << "\n"
                          << "# embeddings: " << result.embeddings.size() << "\n"
                          << "# masks:      " << result.masks.size() << "\n";
            }
            #endif
        }
    }
    return 0;
}
