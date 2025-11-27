#ifndef SAM_ONNX_ROS_SEGMENTATION_HPP_
#define SAM_ONNX_ROS_SEGMENTATION_HPP_

#include "sam_onnx_ros/sam_inference.hpp"

#include <filesystem>

std::tuple<
    std::vector<std::unique_ptr<SAM>>,
    SEG::DL_INIT_PARAM,
    SEG::DL_INIT_PARAM,
    SEG::DL_RESULT,
    std::vector<SEG::DL_RESULT>
>
Initialize(const std::filesystem::path& encoder_filename, const std::filesystem::path& decoder_filename);

void SegmentAnything(
    std::vector<std::unique_ptr<SAM>>& samSegmentors,
    const SEG::DL_INIT_PARAM& params_encoder,
    const SEG::DL_INIT_PARAM& params_decoder,
    const cv::Mat& img,
    std::vector<SEG::DL_RESULT>& resSam,
    SEG::DL_RESULT& res
);

#endif // SAM_ONNX_ROS_SEGMENTATION_HPP_
