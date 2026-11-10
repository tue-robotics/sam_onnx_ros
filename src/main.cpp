#include "sam_onnx_ros/config.hpp"
#include "sam_onnx_ros/dl_types.hpp"
#include "sam_onnx_ros/segmentation.hpp"

#include <cstdint>
#include <exception>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace
{
enum class PromptMode : std::uint8_t
{
    BBOX,
    POINT,
    ROI,
};

bool isSupportedImage(const std::filesystem::path& path)
{
    const std::string extension = path.extension().string();
    return extension == ".jpg" || extension == ".jpeg" || extension == ".png";
}

std::vector<std::filesystem::path> collectInputs(const std::filesystem::path& input_path)
{
    try
    {
        if (!std::filesystem::exists(input_path))
        {
            throw std::invalid_argument("Input path '" + input_path.string() + "' does not exist.");
        }

        std::vector<std::filesystem::path> images;

        if (std::filesystem::is_regular_file(input_path))
        {
            if (isSupportedImage(input_path))
            {
                images.push_back(input_path);
            }
            return images;
        }

        if (!std::filesystem::is_directory(input_path))
        {
            throw std::invalid_argument("Input path '" + input_path.string() + "' is not a regular file or directory.");
        }

        for (const auto& entry : std::filesystem::directory_iterator(input_path))
        {
            if (entry.is_regular_file() && isSupportedImage(entry.path()))
            {
                images.push_back(entry.path());
            }
        }

        std::sort(images.begin(), images.end());
        return images;
    }
    catch (const std::filesystem::filesystem_error& error)
    {
        throw std::invalid_argument("Failed to access input path '" + input_path.string() +
                                    "': " + error.code().message());
    }
}

SEG::Backend parseBackend(const std::string& value)
{
    if (value == "onnx")
    {
        return SEG::Backend::ONNX;
    }
    if (value == "speedsam")
    {
        return SEG::Backend::SPEED_SAM;
    }

    throw std::invalid_argument("Unsupported backend '" + value + "'. Use 'onnx' or 'speedsam'.");
}

PromptMode parsePromptMode(const std::string& value)
{
    if (value == "bbox")
    {
        return PromptMode::BBOX;
    }
    if (value == "point")
    {
        return PromptMode::POINT;
    }
    if (value == "roi")
    {
        return PromptMode::ROI;
    }

    throw std::invalid_argument("Unsupported prompt mode '" + value + "'. Use 'bbox', 'point', or 'roi'.");
}

void printUsage(const char* executable)
{
    std::cerr << "Usage: " << executable << " <encoder_model> <decoder_model> <image_or_dir> "
              << "[--backend=onnx|speedsam] [--prompt=bbox|point|roi] [--logging]" << '\n';
}

int runMain(const std::filesystem::path& encoder_name,
            const std::filesystem::path& decoder_name,
            const std::filesystem::path& input_path,
            SEG::Backend backend,
            PromptMode prompt_mode,
            bool logging)
{
    SamWrapper sam_wrapper;
    SEG::DlInitParam params_encoder;
    SEG::DlInitParam params_decoder;
    std::vector<SEG::DlResult> res_sam;
    SEG::DlResult res;

    std::tie(sam_wrapper, params_encoder, params_decoder, res, res_sam) =
        initialize(encoder_name, decoder_name, backend);

    const auto images = collectInputs(input_path);
    if (images.empty())
    {
        std::cerr << "No supported images found in " << input_path << '\n';
        return 1;
    }

    for (const auto& image_path : images)
    {
        cv::Mat const img = cv::imread(image_path.string());
        if (img.empty())
        {
            std::cerr << "Failed to read image: " << image_path << '\n';
            continue;
        }

        res.boxes.clear();
        res_sam.clear();

        // Populate dummy boxes for decoder as if given by an Object detection Node
        if (prompt_mode == PromptMode::BBOX)
        {
            res.boxes.emplace_back(0, 0, img.cols, img.rows);
        }
        // Or let the user specify a region of interest (ROI)
        else if (prompt_mode == PromptMode::ROI)
        {
            cv::namedWindow("Select ROI", cv::WINDOW_AUTOSIZE);
            cv::Rect const bbox = cv::selectROI("Select ROI", img, false, false);
            cv::destroyWindow("Select ROI");

            if (bbox.width == 0 || bbox.height == 0)
            {
                std::cerr << "No valid bounding box selected. Skipping image." << '\n';
                continue;
            }
            res.boxes.push_back(bbox);
        }
        else
        {
            // Center point fallback logic if extended later
            const int w = std::min(20, img.cols);
            const int h = std::min(20, img.rows);
            const int x = std::max(0, (img.cols / 2) - (w / 2));
            const int y = std::max(0, (img.rows / 2) - (h / 2));
            res.boxes.emplace_back(x, y, w, h);
        }

        segmentAnything(sam_wrapper, params_encoder, params_decoder, img, res_sam, res);

        std::string const mode_str = (backend == SEG::Backend::SPEED_SAM) ? "speedsam" : "onnx";
        std::string prompt_str = " Point prompt";
        if (prompt_mode == PromptMode::BBOX)
        {
            prompt_str = " Bbox prompt";
        }
        else if (prompt_mode == PromptMode::ROI)
        {
            prompt_str = " ROI prompt";
        }
        std::string window_name = "SAM Result (";
        window_name.append(mode_str).append(" -").append(prompt_str).append(")");

        if (!res_sam.empty() && !res_sam.front().masks.empty())
        {
            cv::Mat rendered = img.clone();
            cv::Mat const mask = res_sam.front().masks.front();
            if (!mask.empty())
            {
                cv::Mat color_mask = cv::Mat::zeros(rendered.size(), CV_8UC3);
                color_mask.setTo(cv::Scalar(0, 200, 0), mask);
                cv::addWeighted(rendered, 0.7, color_mask, 0.3, 0, rendered);
                std::vector<std::vector<cv::Point>> contours;
                cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
                cv::drawContours(rendered, contours, -1, cv::Scalar(0, 255, 255), 2);

                cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
                cv::imshow(window_name, rendered);
                std::cout << "Displaying result for " << image_path.filename() << "..." << '\n';
                std::cout << "Press any key on the image window to continue..." << '\n';
                cv::waitKey(0);
                cv::destroyWindow(window_name);
            }
        }

        if (logging)
        {
            for (const auto& result : res_sam)
            {
                std::cout << "Image path:   " << image_path << "\n"
                          << "# boxes:      " << result.boxes.size() << "\n"
                          << "# embeddings: " << result.embeddings.size() << "\n"
                          << "# masks:      " << result.masks.size() << "\n";
            }
        }
    }

    return 0;
}

} // namespace

int main(int argc, char* argv[])
{
    if (argc < 5)
    {
        printUsage(argv[0]);
        return 1;
    }

    const std::filesystem::path encoder_name = argv[1];
    const std::filesystem::path decoder_name = argv[2];
    std::filesystem::path const imgs_path = argv[3];
    bool const logging = argv[4] == std::string("--logging");

    SEG::Backend backend = SEG::Backend::ONNX;
    PromptMode prompt_mode = PromptMode::BBOX;

    try
    {
        for (int i = 4; i < argc; ++i)
        {
            const std::string argument = argv[i];
            if (argument.rfind("--backend=", 0) == 0)
            {
                backend = parseBackend(argument.substr(std::string("--backend=").size()));
            }
            else if (argument.rfind("--prompt=", 0) == 0)
            {
                prompt_mode = parsePromptMode(argument.substr(std::string("--prompt=").size()));
            }
            else if (argument == "--help")
            {
                printUsage(argv[0]);
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
        std::cerr << error.what() << '\n';
        printUsage(argv[0]);
        return 1;
    }

#if !SAM_ONNX_ROS_TENSORRT_ENABLED
    if (backend == SEG::Backend::SPEED_SAM)
    {
        std::cerr
            << "This binary was built without TensorRT support. Install TensorRT and rebuild, or use --backend=onnx."
            << '\n';
        return 2;
    }
#endif

    try
    {
        return runMain(encoder_name, decoder_name, imgs_path, backend, prompt_mode, logging);
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
