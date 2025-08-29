#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#include <tuple>

#include "sam_inference.h"
std::tuple<std::vector<std::unique_ptr<SAM>>, SEG::_DL_INIT_PARAM, SEG::_DL_INIT_PARAM> Initializer();
std::vector<cv::Mat> SegmentAnything(std::vector<std::unique_ptr<SAM>>& samSegmentors, const SEG::_DL_INIT_PARAM& params_encoder, const SEG::_DL_INIT_PARAM& params_decoder, cv::Mat& img);

#endif // SEGMENTATION_H