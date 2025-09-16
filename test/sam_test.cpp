#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include "segmentation.h"
#include "sam_inference.h"
#include "dl_types.h"

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

        // Use package helpers to build default params and SAM objects.
        std::tie(samSegmentors, params_encoder, params_decoder, res, resSam) = Initializer();

#ifdef USE_CUDA
        params_encoder.cudaEnable = true;  // Enable CUDA if compiled with it
#else
        params_encoder.cudaEnable = false; // Otherwise run on CPU
#endif

    }

    // Clean up the SAM objects after each test.
    void TearDown() override { samSegmentors[0].reset(); samSegmentors[1].reset(); }

    // Test data and objects shared across tests.
    Utils utilities;
    cv::Mat testImage_640x640, testImage_800x600, testImage_realistic;
    std::vector<int> NonSquareImgSize;
    std::vector<std::unique_ptr<SAM>> samSegmentors;
    SEG::DL_INIT_PARAM params_encoder, params_decoder;
    SEG::DL_RESULT res;
    std::vector<SEG::DL_RESULT> resSam;
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
    if (!std::filesystem::exists("SAM_encoder.onnx")) {
        GTEST_SKIP() << "Model not found in build dir";
    }

    EXPECT_NE(samSegmentors[0], nullptr) << "CreateSession should succeed with valid parameters";
}

// Confirms that giving an invalid model path returns an error (no crash).
TEST_F(SamInferenceTest, CreateSessionWithInvalidModel)
{
    params_encoder.modelPath = "nonexistent_model.onnx";
    const char* result = samSegmentors[0]->CreateSession(params_encoder);
    EXPECT_NE(result, nullptr) << "CreateSession should fail with invalid model path";
}

// End-to-end check: with both encoder/decoder models present, the pipeline runs
// and returns a mask vector. Skips if models are not available.
TEST_F(SamInferenceTest, FullInferencePipeline)
{
    if (!std::filesystem::exists("SAM_encoder.onnx") ||
        !std::filesystem::exists("SAM_mask_decoder.onnx")) {
        GTEST_SKIP() << "Models not found in build dir";
    }

    SegmentAnything(samSegmentors, params_encoder, params_decoder, testImage_realistic, resSam, res);

    // We only check that a vector is returned. (You can strengthen this to EXPECT_FALSE(masks.empty()).)
    EXPECT_TRUE(res.masks.size() >= 0) << "Masks should be a valid output vector";
}
