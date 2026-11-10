#include "sam_onnx_ros/segmentation.hpp"
#include "sam_onnx_ros/config.hpp"
#include "sam_onnx_ros/dl_types.hpp"
#include "sam_onnx_ros/sam_inference.hpp"
#include <filesystem>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <tuple>
#include <utility>
#include <vector>

#if SAM_ONNX_ROS_TENSORRT_ENABLED
#include <opencv2/core/types.hpp>
#include <speedSam.h>
#else
// Never instantiated, but SamWrapper's std::unique_ptr<SpeedSam> needs a complete type to be destructible
class SpeedSam
{
};
#endif

#include <stdexcept>

SamWrapper::SamWrapper() = default;
SamWrapper::~SamWrapper() = default;
SamWrapper::SamWrapper(SamWrapper&&) noexcept = default;
SamWrapper& SamWrapper::operator=(SamWrapper&&) noexcept = default;

std::tuple<SamWrapper, SEG::DlInitParam, SEG::DlInitParam, SEG::DlResult, std::vector<SEG::DlResult>> initialize(
    const std::filesystem::path& encoder_filename, const std::filesystem::path& decoder_filename, SEG::Backend backend)
{
    SamWrapper sam_wrapper;
    sam_wrapper.backend = backend;

    SEG::DlInitParam params_encoder;
    SEG::DlInitParam params_decoder;
    SEG::DlResult const res;
    std::vector<SEG::DlResult> const res_sam;

    params_encoder.modelPath = encoder_filename.string();
    params_encoder.imgSize = {1024, 1024};
    params_decoder = params_encoder;
    params_decoder.modelType = SEG::ModelType::SAM_SEGMENT_DECODER;
    params_decoder.modelPath = decoder_filename.string();

    if (backend == SEG::Backend::ONNX)
    {
        std::unique_ptr<SAM> sam_segmentor_encoder = std::make_unique<SAM>();
        std::unique_ptr<SAM> sam_segmentor_decoder = std::make_unique<SAM>();

#if defined(SAM_ONNX_ROS_CUDA_ENABLED) && SAM_ONNX_ROS_CUDA_ENABLED
        params_encoder.cudaEnable = true;
        params_decoder.cudaEnable = true;
#else
        params_encoder.cudaEnable = false;
        params_decoder.cudaEnable = false;
#endif

        sam_segmentor_encoder->createSession(params_encoder);
        sam_segmentor_decoder->createSession(params_decoder);

        sam_wrapper.samSegmentors.push_back(std::move(sam_segmentor_encoder));
        sam_wrapper.samSegmentors.push_back(std::move(sam_segmentor_decoder));
    }
#if SAM_ONNX_ROS_TENSORRT_ENABLED
    else if (backend == SEG::Backend::SPEED_SAM)
    {
        sam_wrapper.speedSam = std::make_unique<SpeedSam>(encoder_filename.string(), decoder_filename.string());
    }
#else
    else if (backend == SEG::Backend::SPEED_SAM)
    {
        throw std::runtime_error(
            "[ERROR] Cannot initialize: backend 'speedsam' was requested, but 'sam_onnx_ros' was compiled WITHOUT "
            "TensorRT! Please install TensorRT headers and rebuild.");
    }
#endif

    return {std::move(sam_wrapper), params_encoder, params_decoder, res, res_sam};
}

void segmentAnything(SamWrapper& samWrapper,
                     const SEG::DlInitParam& params_encoder,
                     const SEG::DlInitParam& params_decoder,
                     const cv::Mat& img,
                     std::vector<SEG::DlResult>& resSam,
                     SEG::DlResult& res)
{
    if (samWrapper.backend == SEG::Backend::ONNX)
    {
        if (samWrapper.samSegmentors.size() < 2 || !samWrapper.samSegmentors[0] || !samWrapper.samSegmentors[1])
        {
            throw std::runtime_error(
                "segmentAnything: ONNX backend requires 2 initialized SAM sessions (encoder + decoder).");
        }

        SEG::ModelType model_type_ref = params_encoder.modelType;
        samWrapper.samSegmentors[0]->runSession(img, resSam, model_type_ref, res);

        model_type_ref = params_decoder.modelType;
        samWrapper.samSegmentors[1]->runSession(img, resSam, model_type_ref, res);
    }
#if SAM_ONNX_ROS_TENSORRT_ENABLED
    else if (samWrapper.backend == SEG::Backend::SPEED_SAM)
    {
        // Mimic the exact behaviour of Onnx Pipeline: encode then decode per box.
        // It outputs masks bounding boxes inside resSam.
        // For SpeedSam, we encode the image once, then we just loop over `res.boxes` and push output into `resSam[0]`.

        res.masks.clear();

        // setTemplateImage takes a non-const reference; a cv::Mat copy only copies the header, not the pixels
        cv::Mat template_image = img;
        samWrapper.speedSam->setTemplateImage(template_image);

        for (const auto& box : res.boxes)
        {
            std::vector<cv::Point> const bbox_points = {cv::Point(box.x, box.y),
                                                        cv::Point(box.x + box.width, box.y + box.height)};
            cv::Mat const mask = samWrapper.speedSam->predictFromTemplate(bbox_points, {2.0F, 3.0F});

            res.masks.push_back(mask);
        }
        resSam.push_back(res);
    }
#else
    else if (samWrapper.backend == SEG::Backend::SPEED_SAM)
    {
        // Fail loudly rather than doing 0ms silent returns
        throw std::runtime_error(
            "[ERROR] segmentAnything: backend 'speedsam' was requested, but 'sam_onnx_ros' was compiled WITHOUT "
            "TensorRT! Please install TensorRT headers and rebuild.");
    }
#endif
}
