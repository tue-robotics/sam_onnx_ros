#include "sam_onnx_ros/config.hpp"
#include "sam_onnx_ros/segmentation.hpp"

std::tuple<
    SamWrapper,
    SEG::DL_INIT_PARAM,
    SEG::DL_INIT_PARAM,
    SEG::DL_RESULT,
    std::vector<SEG::DL_RESULT>
>
Initialize(const std::filesystem::path& encoder_filename, const std::filesystem::path& decoder_filename, SEG::Backend backend)
{
    SamWrapper samWrapper;
    samWrapper.backend = backend;

    SEG::DL_INIT_PARAM params_encoder;
    SEG::DL_INIT_PARAM params_decoder;
    SEG::DL_RESULT res;
    std::vector<SEG::DL_RESULT> resSam;

    params_encoder.modelPath = encoder_filename.string();
    params_encoder.imgSize = {1024, 1024};
    params_decoder = params_encoder;
    params_decoder.modelType = SEG::SAM_SEGMENT_DECODER;
    params_decoder.modelPath = decoder_filename.string();

    if (backend == SEG::Backend::kOnnx)
    {
        std::unique_ptr<SAM> samSegmentorEncoder = std::make_unique<SAM>();
        std::unique_ptr<SAM> samSegmentorDecoder = std::make_unique<SAM>();

        #if defined(SAM_ONNX_ROS_CUDA_ENABLED) && SAM_ONNX_ROS_CUDA_ENABLED
        params_encoder.cudaEnable = true;
        params_decoder.cudaEnable = true;
        #else
        params_encoder.cudaEnable = false;
        params_decoder.cudaEnable = false;
        #endif

        samSegmentorEncoder->CreateSession(params_encoder);
        samSegmentorDecoder->CreateSession(params_decoder);

        samWrapper.samSegmentors.push_back(std::move(samSegmentorEncoder));
        samWrapper.samSegmentors.push_back(std::move(samSegmentorDecoder));
    }
#if SAM_ONNX_ROS_TENSORRT_ENABLED
    else if (backend == SEG::Backend::kSpeedSam)
    {
        samWrapper.speedSam = std::make_unique<SpeedSam>(encoder_filename.string(), decoder_filename.string());
    }
#else
    else if (backend == SEG::Backend::kSpeedSam)
    {
        throw std::runtime_error("[ERROR] Cannot Initialize: backend 'speedsam' was requested, but 'sam_onnx_ros' was compiled WITHOUT TensorRT! Please install TensorRT headers and rebuild.");
    }
#endif

    return {std::move(samWrapper), params_encoder, params_decoder, res, resSam};
}

void SegmentAnything(SamWrapper& samWrapper,
                     const SEG::DL_INIT_PARAM& params_encoder,
                     const SEG::DL_INIT_PARAM& params_decoder,
                     const cv::Mat& img,
                     std::vector<SEG::DL_RESULT>& resSam,
                     SEG::DL_RESULT& res)
{
    if (samWrapper.backend == SEG::Backend::kOnnx)
    {
        SEG::MODEL_TYPE modelTypeRef = params_encoder.modelType;
        samWrapper.samSegmentors[0]->RunSession(img, resSam, modelTypeRef, res);

        modelTypeRef = params_decoder.modelType;
        samWrapper.samSegmentors[1]->RunSession(img, resSam, modelTypeRef, res);
    }
#if SAM_ONNX_ROS_TENSORRT_ENABLED
    else if (samWrapper.backend == SEG::Backend::kSpeedSam)
    {
        // Mimic the exact behaviour of Onnx Pipeline: encode then decode per box.
        // It outputs masks bounding boxes inside resSam.
        // For SpeedSam, we encode the image once, then we just loop over `res.boxes` and push output into `resSam[0]`.

        res.masks.clear();

        samWrapper.speedSam->setTemplateImage(const_cast<cv::Mat&>(img));

        for (const auto& box : res.boxes)
        {
            std::vector<cv::Point> bboxPoints = {
                cv::Point(box.x, box.y),
                cv::Point(box.x + box.width, box.y + box.height)
            };
            cv::Mat mask = samWrapper.speedSam->predictFromTemplate(bboxPoints, {2.0f, 3.0f});

            res.masks.push_back(mask);
        }
        resSam.push_back(res);
    }
#else
    else if (samWrapper.backend == SEG::Backend::kSpeedSam)
    {
        // Fail loudly rather than doing 0ms silent returns
        throw std::runtime_error("[ERROR] SegmentAnything: backend 'speedsam' was requested, but 'sam_onnx_ros' was compiled WITHOUT TensorRT! Please install TensorRT headers and rebuild.");
    }
#endif
}
