#ifndef SAM_ONNX_ROS_SEGMENTATION_HPP_
#define SAM_ONNX_ROS_SEGMENTATION_HPP_

#include "sam_onnx_ros/config.hpp"
#include "sam_onnx_ros/sam_inference.hpp"

#include <filesystem>
#include <memory>
#include <vector>

namespace SEG
{
enum class Backend
{
    ONNX,
    SPEED_SAM,
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

std::tuple<SamWrapper, SEG::DlInitParam, SEG::DlInitParam, SEG::DlResult, std::vector<SEG::DlResult>> initialize(
    const std::filesystem::path& encoder_filename, const std::filesystem::path& decoder_filename, SEG::Backend backend);

void segmentAnything(SamWrapper& samSegmentors,
                     const SEG::DlInitParam& params_encoder,
                     const SEG::DlInitParam& params_decoder,
                     const cv::Mat& img,
                     std::vector<SEG::DlResult>& resSam,
                     SEG::DlResult& res);

#endif // SAM_ONNX_ROS_SEGMENTATION_HPP_
