#ifndef SAM_ONNX_ROS_SEGMENTATION_HPP_
#define SAM_ONNX_ROS_SEGMENTATION_HPP_

#include "sam_onnx_ros/sam_inference.hpp"
#include "sam_onnx_ros/config.hpp"

#if SAM_ONNX_ROS_TENSORRT_ENABLED
#include <speedSam.h>
#endif

#include <filesystem>
#include <memory>
#include <vector>

namespace SEG
{
    enum class Backend
    {
        kOnnx,
        kSpeedSam,
    };
}

class SamWrapper
{
public:
    SEG::Backend backend;
    std::vector<std::unique_ptr<SAM>> samSegmentors;
#if SAM_ONNX_ROS_TENSORRT_ENABLED
    std::unique_ptr<SpeedSam> speedSam;
#endif

    SamWrapper() : backend(SEG::Backend::kOnnx) {}
};

std::tuple<
    SamWrapper,
    SEG::DL_INIT_PARAM,
    SEG::DL_INIT_PARAM,
    SEG::DL_RESULT,
    std::vector<SEG::DL_RESULT>
>
Initialize(const std::filesystem::path& encoder_filename, const std::filesystem::path& decoder_filename, SEG::Backend backend);

void SegmentAnything(
    SamWrapper& samSegmentors,
    const SEG::DL_INIT_PARAM& params_encoder,
    const SEG::DL_INIT_PARAM& params_decoder,
    const cv::Mat& img,
    std::vector<SEG::DL_RESULT>& resSam,
    SEG::DL_RESULT& res
);

#endif // SAM_ONNX_ROS_SEGMENTATION_HPP_
