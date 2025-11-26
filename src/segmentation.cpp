#include "sam_onnx_ros/config.hpp"
#include "sam_onnx_ros/segmentation.hpp"

std::tuple<
    std::vector<std::unique_ptr<SAM>>,
    SEG::DL_INIT_PARAM,
    SEG::DL_INIT_PARAM,
    SEG::DL_RESULT,
    std::vector<SEG::DL_RESULT>
>
Initialize(const std::filesystem::path& encoder_filename, const std::filesystem::path& decoder_filename)
{
    std::vector<std::unique_ptr<SAM>> samSegmentors;
    samSegmentors.push_back(std::make_unique<SAM>());
    samSegmentors.push_back(std::make_unique<SAM>());

    std::unique_ptr<SAM> samSegmentorEncoder = std::make_unique<SAM>();
    std::unique_ptr<SAM> samSegmentorDecoder = std::make_unique<SAM>();
    SEG::DL_INIT_PARAM params_encoder;
    SEG::DL_INIT_PARAM params_decoder;
    SEG::DL_RESULT res;
    std::vector<SEG::DL_RESULT> resSam;
    params_encoder.modelPath = encoder_filename;
    params_encoder.imgSize = {1024, 1024};

    params_decoder = params_encoder;
    params_decoder.modelType = SEG::SAM_SEGMENT_DECODER;
    params_decoder.modelPath = decoder_filename;

    #if defined(SAM_ONNX_ROS_CUDA_ENABLED) && SAM_ONNX_ROS_CUDA_ENABLED
    params_encoder.cudaEnable = true;
    params_decoder.cudaEnable = true;

    #else
    params_encoder.cudaEnable = false;
    params_decoder.cudaEnable = false;
    #endif

    samSegmentorEncoder->CreateSession(params_encoder);
    samSegmentorDecoder->CreateSession(params_decoder);

    samSegmentors[0] = std::move(samSegmentorEncoder);
    samSegmentors[1] = std::move(samSegmentorDecoder);

    return {std::move(samSegmentors), params_encoder, params_decoder, res, resSam};
}

void SegmentAnything(std::vector<std::unique_ptr<SAM>>& samSegmentors,
                     const SEG::DL_INIT_PARAM& params_encoder,
                     const SEG::DL_INIT_PARAM& params_decoder,
                     const cv::Mat& img,
                     std::vector<SEG::DL_RESULT>& resSam,
                     SEG::DL_RESULT& res)
{

    SEG::MODEL_TYPE modelTypeRef = params_encoder.modelType;
    samSegmentors[0]->RunSession(img, resSam, modelTypeRef, res);

    modelTypeRef = params_decoder.modelType;
    samSegmentors[1]->RunSession(img, resSam, modelTypeRef, res);

    // return std::move(res.masks);
}
