#include "sam_onnx_ros/config.hpp"
#include "sam_onnx_ros/segmentation.hpp"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#if SAM_ONNX_ROS_TENSORRT_ENABLED
#include "speedSam.h"
#include "utils.h"
#endif

namespace
{
enum class PromptMode
{
    kBbox,
    kPoint,
    kRoi,
};

// #define LOGGING

bool IsSupportedImage(const std::filesystem::path& path)
{
    const std::string extension = path.extension().string();
    return extension == ".jpg" || extension == ".jpeg" || extension == ".png";
}

std::vector<std::filesystem::path> CollectInputs(const std::filesystem::path& input_path)
{
    std::vector<std::filesystem::path> images;

    if (std::filesystem::is_regular_file(input_path))
    {
        if (IsSupportedImage(input_path))
        {
            images.push_back(input_path);
        }
        return images;
    }

    for (const auto& entry : std::filesystem::directory_iterator(input_path))
    {
        if (entry.is_regular_file() && IsSupportedImage(entry.path()))
        {
            images.push_back(entry.path());
        }
    }

    std::sort(images.begin(), images.end());
    return images;
}

SEG::Backend ParseBackend(const std::string& value)
{
    if (value == "onnx")
    {
        return SEG::Backend::kOnnx;
    }
    if (value == "speedsam")
    {
        return SEG::Backend::kSpeedSam;
    }

    throw std::invalid_argument("Unsupported backend '" + value + "'. Use 'onnx' or 'speedsam'.");
}

PromptMode ParsePromptMode(const std::string& value)
{
    if (value == "bbox")
    {
        return PromptMode::kBbox;
    }
    if (value == "point")
    {
        return PromptMode::kPoint;
    }
    if (value == "roi")
    {
        return PromptMode::kRoi;
    }

    throw std::invalid_argument("Unsupported prompt mode '" + value + "'. Use 'bbox', 'point', or 'roi'.");
}

void PrintUsage(const char* executable)
{
    std::cerr << "Usage: " << executable << " <encoder_model> <decoder_model> <image_or_dir> "
              << "[--backend=onnx|speedsam] [--prompt=bbox|point|roi]" << std::endl;
}

int RunMain(const std::filesystem::path& encoder_name,
            const std::filesystem::path& decoder_name,
            const std::filesystem::path& input_path,
            SEG::Backend backend,
            PromptMode prompt_mode)
{
    SamWrapper samWrapper;
    SEG::DL_INIT_PARAM params_encoder;
    SEG::DL_INIT_PARAM params_decoder;
    std::vector<SEG::DL_RESULT> resSam;
    SEG::DL_RESULT res;

    std::tie(samWrapper, params_encoder, params_decoder, res, resSam) = Initialize(encoder_name, decoder_name, backend);

    const auto images = CollectInputs(input_path);
    if (images.empty())
    {
        std::cerr << "No supported images found in " << input_path << std::endl;
        return 1;
    }

    for (const auto& image_path : images)
    {
        cv::Mat img = cv::imread(image_path.string());
        if (img.empty())
        {
            std::cerr << "Failed to read image: " << image_path << std::endl;
            continue;
        }


        res.boxes.clear();
        resSam.clear();

        // Populate dummy boxes for decoder as if given by an Object detection Node
        if (prompt_mode == PromptMode::kBbox)
        {
            res.boxes.push_back(cv::Rect(0, 0, std::max(img.cols - 1, 0), std::max(img.rows - 1, 0)));
        }
        // Or let the user specify a region of interest (ROI)
        else if (prompt_mode == PromptMode::kRoi)
        {
            cv::namedWindow("Select ROI", cv::WINDOW_AUTOSIZE);
            cv::Rect bbox = cv::selectROI("Select ROI", img, false, false);
            cv::destroyWindow("Select ROI");

            if (bbox.width == 0 || bbox.height == 0)
            {
                std::cerr << "No valid bounding box selected. Skipping image." << std::endl;
                continue;
            }
            res.boxes.push_back(bbox);
        }
        else
        {
            // Center point fallback logic if extended later
            res.boxes.push_back(cv::Rect(img.cols / 2 - 10, img.rows / 2 - 10, 20, 20));
        }

        SegmentAnything(samWrapper, params_encoder, params_decoder, img, resSam, res);

        std::string modeStr = (backend == SEG::Backend::kSpeedSam) ? "speedsam" : "onnx";
        std::string promptStr = (prompt_mode == PromptMode::kBbox) ? " Bbox prompt" :
                                (prompt_mode == PromptMode::kRoi)  ? " ROI prompt" : " Point prompt";
        std::string windowName = "SAM Result (" + modeStr + " -" + promptStr + ")";

        if (!resSam.empty() && !resSam.front().masks.empty())
        {
            cv::Mat rendered = img.clone();
            cv::Mat mask = resSam.front().masks.front();
            if (!mask.empty()) {
                cv::Mat colorMask = cv::Mat::zeros(rendered.size(), CV_8UC3);
                colorMask.setTo(cv::Scalar(0, 200, 0), mask);
                cv::addWeighted(rendered, 0.7, colorMask, 0.3, 0, rendered);
                std::vector<std::vector<cv::Point>> contours;
                cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
                cv::drawContours(rendered, contours, -1, cv::Scalar(0, 255, 255), 2);

                cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
                cv::imshow(windowName, rendered);
                std::cout << "Displaying result for " << image_path.filename() << "..." << std::endl;
                std::cout << "Press any key on the image window to continue..." << std::endl;
                cv::waitKey(0);
                cv::destroyWindow(windowName);
            }
        }

#ifdef LOGGING
        for (const auto& result : resSam)
        {
            std::cout << "Image path:   " << image_path << "\n"
                      << "# boxes:      " << result.boxes.size() << "\n"
                      << "# embeddings: " << result.embeddings.size() << "\n"
                      << "# masks:      " << result.masks.size() << "\n";
        }
#endif
    }

    return 0;
}

} // namespace

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        PrintUsage(argv[0]);
        return 1;
    }

    const std::filesystem::path encoder_name = argv[1];
    const std::filesystem::path decoder_name = argv[2];
    std::filesystem::path imgs_path = argv[3];

    SEG::Backend backend = SEG::Backend::kOnnx;
    PromptMode prompt_mode = PromptMode::kBbox;

    try
    {
        for (int i = 4; i < argc; ++i)
        {
            const std::string argument = argv[i];
            if (argument.rfind("--backend=", 0) == 0)
            {
                backend = ParseBackend(argument.substr(std::string("--backend=").size()));
            }
            else if (argument.rfind("--prompt=", 0) == 0)
            {
                prompt_mode = ParsePromptMode(argument.substr(std::string("--prompt=").size()));
            }
            else if (argument == "--help")
            {
                PrintUsage(argv[0]);
                return 0;
            }
            else
            {
                throw std::invalid_argument("Unknown argument '" + argument + "'.");
            }
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << std::endl;
        PrintUsage(argv[0]);
        return 1;
    }

#if !SAM_ONNX_ROS_TENSORRT_ENABLED
    if (backend == SEG::Backend::kSpeedSam)
    {
        std::cerr << "This binary was built without TensorRT support. Install TensorRT and rebuild, or use --backend=onnx." << std::endl;
        return 2;
    }
#endif

    return RunMain(encoder_name, decoder_name, imgs_path, backend, prompt_mode);
}
