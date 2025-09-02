#include "segmentation.h"
#include "sam_inference.h"
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "dl_types.h"
#include "utils.h"
#include <filesystem>

class SamInferenceTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create test images with different characteristics
        testImage_640x640 = cv::Mat::ones(640, 640, CV_8UC3) * 255;
        testImage_800x600 = cv::Mat::ones(600, 800, CV_8UC3) * 128;

        // Create a more realistic test image with some patterns
        testImage_realistic = cv::Mat(640, 640, CV_8UC3);
        cv::randu(testImage_realistic, cv::Scalar(0,0,0), cv::Scalar(255,255,255));

        // Setup common parameters
        NonSquareImgSize = { testImage_800x600.cols, testImage_800x600.rows };

        sam = std::make_unique<SAM>();
        params.rectConfidenceThreshold = 0.1f;
        params.iouThreshold = 0.5f;
        params.imgSize = {1024, 1024};
        params.modelType = SEG::SAM_SEGMENT_ENCODER;
        params.modelPath = "SAM_encoder.onnx"; // copied to build/ by CMake
#ifdef USE_CUDA
        params.cudaEnable = true;
#else
        params.cudaEnable = false;
#endif
    }

    void TearDown() override { sam.reset(); }

    // Test data
    Utils utilities;
    cv::Mat testImage_640x640, testImage_800x600, testImage_realistic;
    SEG::DL_INIT_PARAM params;
    std::unique_ptr<SAM> sam;
    std::vector<int> NonSquareImgSize;
};



TEST_F(SamInferenceTest, ObjectCreation)
{
    EXPECT_NO_THROW({
        SAM localSam;
    });
}

TEST_F(SamInferenceTest, PreProcessSquareImage)
{
    cv::Mat processedImg;
    const char* result = utilities.PreProcess(testImage_640x640, params.imgSize, processedImg);

    EXPECT_EQ(result, nullptr) << "PreProcess should succeed";
    EXPECT_EQ(processedImg.size(), cv::Size(1024, 1024)) << "Output should be letterboxed to 1024x1024";
    EXPECT_FALSE(processedImg.empty()) << "Processed image should not be empty";
}

TEST_F(SamInferenceTest, PreProcessRectangularImage)
{
    cv::Mat processedImg;
    const char* result = utilities.PreProcess(testImage_800x600, NonSquareImgSize, processedImg);

    EXPECT_EQ(result, nullptr) << "PreProcess should succeed";
    EXPECT_EQ(processedImg.size(), cv::Size(800, 600)) << "Output should be letterboxed to 800x600";
    EXPECT_FALSE(processedImg.empty()) << "Processed image should not be empty";
}

TEST_F(SamInferenceTest, CreateSessionWithValidModel)
{
    if (!std::filesystem::exists("SAM_encoder.onnx")) {
        GTEST_SKIP() << "Model not found in build dir";
    }
    const char* result = sam->CreateSession(params);
    EXPECT_EQ(result, nullptr) << "CreateSession should succeed with valid parameters";
}

TEST_F(SamInferenceTest, CreateSessionWithInvalidModel)
{
    params.modelPath = "nonexistent_model.onnx";
    const char* result = sam->CreateSession(params);
    EXPECT_NE(result, nullptr) << "CreateSession should fail with invalid model path";
}

TEST_F(SamInferenceTest, FullInferencePipeline)
{
    if (!std::filesystem::exists("SAM_encoder.onnx") ||
        !std::filesystem::exists("SAM_mask_decoder.onnx")) {
        GTEST_SKIP() << "Models not found in build dir";
    }

    // Use the package Initializer/SegmentAnything for the full pipeline
    std::vector<std::unique_ptr<SAM>> samSegmentors;
    SEG::DL_INIT_PARAM params_encoder, params_decoder;
    std::tie(samSegmentors, params_encoder, params_decoder) = Initializer();

    auto masks = SegmentAnything(samSegmentors, params_encoder, params_decoder, testImage_realistic);
    EXPECT_TRUE(masks.size() >= 0) << "Masks should be a valid output vector";
}

// Run all tests
int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}