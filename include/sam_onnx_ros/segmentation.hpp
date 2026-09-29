#ifndef SAM_ONNX_ROS_SEGMENTATION_HPP_
#define SAM_ONNX_ROS_SEGMENTATION_HPP_

#include "sam_onnx_ros/sam_inference.hpp"
#include "sam_onnx_ros/config.hpp"

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

// Forward declared to keep the sam_trt_lib headers out of the public include path
class SpeedSam;

class SamWrapper
{
public:
    SEG::Backend backend;
    std::vector<std::unique_ptr<SAM>> samSegmentors;
    // Only set for the SpeedSAM backend, which requires TensorRT support at compile time
    std::unique_ptr<SpeedSam> speedSam;

    // Defined in segmentation.cpp, where SpeedSam is a complete type
    SamWrapper();
    ~SamWrapper();
    SamWrapper(SamWrapper&&) noexcept;
    SamWrapper& operator=(SamWrapper&&) noexcept;
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
