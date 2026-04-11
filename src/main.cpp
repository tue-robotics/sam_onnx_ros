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
enum class Backend
{
    kOnnx,
    kSpeedSam,
};

enum class PromptMode
{
    kBbox,
    kPoint,
};

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

Backend ParseBackend(const std::string& value)
{
    if (value == "onnx")
    {
        return Backend::kOnnx;
    }
    if (value == "speedsam")
    {
        return Backend::kSpeedSam;
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

    throw std::invalid_argument("Unsupported prompt mode '" + value + "'. Use 'bbox' or 'point'.");
}

void PrintUsage(const char* executable)
{
    std::cerr << "Usage: " << executable << " <encoder_model> <decoder_model> <image_or_dir> "
              << "[--backend=onnx|speedsam] [--prompt=bbox|point]" << std::endl;
}

int RunOnnxMain(const std::filesystem::path& encoder_name,
                const std::filesystem::path& decoder_name,
                const std::filesystem::path& input_path)
{
    std::vector<std::unique_ptr<SAM>> samSegmentors;
    SEG::DL_INIT_PARAM params_encoder;
    SEG::DL_INIT_PARAM params_decoder;
    std::vector<SEG::DL_RESULT> resSam;
    SEG::DL_RESULT res;

    std::tie(samSegmentors, params_encoder, params_decoder, res, resSam) = Initialize(encoder_name, decoder_name);

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

        SegmentAnything(samSegmentors, params_encoder, params_decoder, img, resSam, res);
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

#if SAM_ONNX_ROS_TENSORRT_ENABLED
int RunSpeedSamMain(const std::filesystem::path& encoder_name,
                    const std::filesystem::path& decoder_name,
                    const std::filesystem::path& input_path,
                    PromptMode prompt_mode)
{
    const auto images = CollectInputs(input_path);
    if (images.empty())
    {
        std::cerr << "No supported images found in " << input_path << std::endl;
        return 1;
    }

    SpeedSam speed_sam(encoder_name.string(), decoder_name.string());

    for (const auto& image_path : images)
    {
        cv::Mat image = cv::imread(image_path.string());
        if (image.empty())
        {
            std::cerr << "Failed to read image: " << image_path << std::endl;
            continue;
        }

        const std::filesystem::path output_path =
            image_path.parent_path() /
            (image_path.stem().string() +
             (prompt_mode == PromptMode::kBbox ? "_speedsam_bbox_mask.png" : "_speedsam_point_mask.png"));

        cv::Mat mask;
        if (prompt_mode == PromptMode::kPoint)
        {
            const cv::Point center(image.cols / 2, image.rows / 2);
            mask = speed_sam.predict(image, {center}, {1.0f});
        }
        else
        {
            const std::vector<cv::Point> bboxPoints = {
                cv::Point(0, 0),
                cv::Point(std::max(image.cols - 1, 0), std::max(image.rows - 1, 0))
            };
            mask = speed_sam.predict(image, bboxPoints, {2.0f, 3.0f});
        }

        if (mask.empty())
        {
            std::cerr << "SpeedSAM produced an empty mask for " << image_path << std::endl;
            continue;
        }

        cv::Mat rendered = image.clone();
        overlay(rendered, mask);
        cv::imwrite(output_path.string(), rendered);
        std::cout << "Saved SpeedSAM output to " << output_path << std::endl;
    }

    return 0;
}
#endif
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

    Backend backend = Backend::kOnnx;
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

    if (backend == Backend::kOnnx)
    {
        return RunOnnxMain(encoder_name, decoder_name, imgs_path);
    }

#if SAM_ONNX_ROS_TENSORRT_ENABLED
    return RunSpeedSamMain(encoder_name, decoder_name, imgs_path, prompt_mode);
#else
    (void)prompt_mode;
    std::cerr << "This binary was built without TensorRT support. Install TensorRT and rebuild, or use --backend=onnx." << std::endl;
    return 2;
#endif
}
