#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#include <tuple>

#include "sam_inference.h"
std::tuple<std::vector<std::unique_ptr<SAM>>, SEG::_DL_INIT_PARAM, SEG::_DL_INIT_PARAM, SEG::DL_RESULT, std::vector<SEG::DL_RESULT>> Initializer();
void SegmentAnything(std::vector<std::unique_ptr<SAM>>& samSegmentors, const SEG::_DL_INIT_PARAM& params_encoder, const SEG::_DL_INIT_PARAM& params_decoder, const cv::Mat& img,
std::vector<SEG::DL_RESULT> &resSam,
  SEG::DL_RESULT &res);

#endif // SEGMENTATION_H
