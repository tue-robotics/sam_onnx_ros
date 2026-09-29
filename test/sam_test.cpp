#include "sam_onnx_ros/dl_types.hpp"
#include "sam_onnx_ros/sam_inference.hpp"
#include "sam_onnx_ros/segmentation.hpp"

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include <filesystem>
#include <string>

#if SAM_ONNX_ROS_TENSORRT_ENABLED
#include <cuda_runtime_api.h>
#endif

// This file contains higher-level (integration-ish) tests.
// They cover object/session creation and a full pipeline run using synthetic images.
// These tests may require the .onnx model files to be present next to the binary or in a known dir.

class SamInferenceTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create simple synthetic images:
        // - a white 640x640 (square)
        // - a gray 800x600 (non-square)
        testImage_640x640 = cv::Mat::ones(640, 640, CV_8UC3) * 255;
        testImage_800x600 = cv::Mat::ones(600, 800, CV_8UC3) * 128;

        // A "random noise" image to simulate realistic content for end-to-end checks.
        testImage_realistic = cv::Mat(640, 640, CV_8UC3);
        cv::randu(testImage_realistic, cv::Scalar(0,0,0), cv::Scalar(255,255,255));

        // Cache non-square size for preprocessing helpers.
        NonSquareImgSize = { testImage_800x600.cols, testImage_800x600.rows };

        const std::filesystem::path encoder_model = "./SAM_encoder.onnx";
        const std::filesystem::path decoder_model = "./SAM_mask_decoder.onnx";

        if (!std::filesystem::exists(encoder_model) || !std::filesystem::exists(decoder_model))
        {
            models_available_ = false;
            missing_models_reason_ = "Required models not found in working directory: './SAM_encoder.onnx' and './SAM_mask_decoder.onnx'.";
            return;
        }

        try
        {
            // Use package helpers to build default params and SAM objects.
            std::tie(samWrapper, params_encoder, params_decoder, res, resSam) =
                Initialize(encoder_model, decoder_model, SEG::Backend::kOnnx);
            models_available_ = true;
        }
        catch (const std::exception& e)
        {
            models_available_ = false;
            missing_models_reason_ = std::string("Model initialization failed: ") + e.what();
        }

    }

    // Clean up the SAM objects after each test.
    void TearDown() override {
        samWrapper = SamWrapper();
    }

    // Test data and objects shared across tests.
    Utils utilities;
    cv::Mat testImage_640x640, testImage_800x600, testImage_realistic;
    std::vector<int> NonSquareImgSize;
    SamWrapper samWrapper;
    SEG::DL_INIT_PARAM params_encoder, params_decoder;
    SEG::DL_RESULT res;
    std::vector<SEG::DL_RESULT> resSam;
    bool models_available_ = false;
    std::string missing_models_reason_;
};

// Simple smoke test: we can construct a SAM object without throwing.
TEST_F(SamInferenceTest, ObjectCreation)
{
    EXPECT_NO_THROW({
        SAM localSam;
    });
}

// Confirms that with a present encoder model we can initialize a session.
// Skips if the model file is not available.
TEST_F(SamInferenceTest, CreateSessionWithValidModel)
{
    // GTEST_SKIP only returns from the current function, so it must be called from the test body
    if (!models_available_)
    {
        GTEST_SKIP() << missing_models_reason_;
    }

    EXPECT_EQ(samWrapper.samSegmentors.size(), 2u)
        << "Initialize should create both encoder and decoder sessions";
    ASSERT_TRUE(samWrapper.samSegmentors[0]);
    ASSERT_TRUE(samWrapper.samSegmentors[1]);
}

// Confirms that giving an invalid model path returns an error (no crash).
TEST_F(SamInferenceTest, CreateSessionWithInvalidModel)
{
    params_encoder.modelPath = "nonexistent_model.onnx";
    if (samWrapper.samSegmentors.empty()) {
        samWrapper.samSegmentors.push_back(std::make_unique<SAM>());
    }
    EXPECT_THROW(samWrapper.samSegmentors[0]->CreateSession(params_encoder), std::runtime_error)
        << "CreateSession should throw an exception with invalid model path";
}

// End-to-end check: with both encoder/decoder models present, the pipeline runs
// and returns a mask vector. Skips if models are not available.
TEST_F(SamInferenceTest, FullInferencePipeline)
{
    // GTEST_SKIP only returns from the current function, so it must be called from the test body
    if (!models_available_)
    {
        GTEST_SKIP() << missing_models_reason_;
    }

    SegmentAnything(samWrapper, params_encoder, params_decoder, testImage_realistic, resSam, res);
}

#if !SAM_ONNX_ROS_TENSORRT_ENABLED
// Without TensorRT support, requesting the SpeedSAM backend must fail loudly instead of silently doing nothing.
// Needs no models or GPU.
TEST(SpeedSamBackendTest, InitializeThrowsWithoutTensorRT)
{
    EXPECT_THROW(Initialize("./SAM_encoder.onnx", "./SAM_mask_decoder.onnx", SEG::Backend::kSpeedSam), std::runtime_error)
        << "Initialize should throw when the SpeedSAM backend is requested without TensorRT support";
}

TEST(SpeedSamBackendTest, SegmentAnythingThrowsWithoutTensorRT)
{
    SamWrapper samWrapper;
    samWrapper.backend = SEG::Backend::kSpeedSam;
    SEG::DL_INIT_PARAM params_encoder, params_decoder;
    SEG::DL_RESULT res;
    std::vector<SEG::DL_RESULT> resSam;
    const cv::Mat img = cv::Mat::zeros(640, 640, CV_8UC3);

    EXPECT_THROW(SegmentAnything(samWrapper, params_encoder, params_decoder, img, resSam, res), std::runtime_error)
        << "SegmentAnything should throw when the SpeedSAM backend is requested without TensorRT support";
}
#else
// End-to-end check of the SpeedSAM backend: one box prompt gives one mask of the input image size.
// Skips if no CUDA device or no models are available.
TEST(SpeedSamBackendTest, FullInferencePipeline)
{
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
    {
        GTEST_SKIP() << "No CUDA device available";
    }

    const std::filesystem::path encoder_model = "./SAM_encoder.onnx";
    const std::filesystem::path decoder_model = "./SAM_mask_decoder.onnx";
    if (!std::filesystem::exists(encoder_model) || !std::filesystem::exists(decoder_model))
    {
        GTEST_SKIP() << "Required models not found in working directory: './SAM_encoder.onnx' and './SAM_mask_decoder.onnx'.";
    }

    auto [samWrapper, params_encoder, params_decoder, res, resSam] =
        Initialize(encoder_model, decoder_model, SEG::Backend::kSpeedSam);
    ASSERT_TRUE(samWrapper.speedSam) << "Initialize should create the SpeedSAM backend";

    cv::Mat img(600, 800, CV_8UC3);
    cv::randu(img, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
    res.boxes.push_back(cv::Rect(200, 150, 400, 300));

    SegmentAnything(samWrapper, params_encoder, params_decoder, img, resSam, res);

    ASSERT_EQ(res.masks.size(), 1u) << "SpeedSAM should return one mask per box";
    EXPECT_EQ(res.masks[0].size(), img.size());
    EXPECT_EQ(res.masks[0].type(), CV_8UC1);
}
#endif

// Run all tests
int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
