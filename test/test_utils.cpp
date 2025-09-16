#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "utils.h"

// This file contains small, focused unit tests for Utils.
// We verify image preprocessing (channel conversion, aspect-preserving resize, padding)
// and coordinate scaling to match preprocessing.

// Lightweight fixture: gives each test a fresh Utils instance.
class UtilsTest : public ::testing::Test {
protected:
    Utils u;
};

// Checks that a grayscale (1-channel) image is converted to RGB (3-channel)
// and the output image is exactly the requested target size (letterboxed).
TEST_F(UtilsTest, GrayscaleToRGBKeepsSize) {
    cv::Mat gray = cv::Mat::zeros(300, 500, CV_8UC1);
    cv::Mat out;
    std::vector<int> target{1024, 1024};

    // Call PreProcess and expect no error.
    const char* err = u.PreProcess(gray, target, out);
    ASSERT_EQ(err, nullptr);

    // After preprocessing, we must have 3 channels (RGB).
    EXPECT_EQ(out.channels(), 3);

    // The letterboxed output must match the target canvas size.
    EXPECT_EQ(out.size(), cv::Size(target[0], target[1]));
}

// Verifies three things:
// 1) Aspect ratio is preserved when resizing to the target.
// 2) The resized image is placed at the top-left (0,0).
// 3) The padding area is zero (black).
TEST_F(UtilsTest, PreprocessTopLeftPaddingAndAspect) {
    const cv::Scalar fill(10, 20, 30); // Input color in BGR
    cv::Mat img(720, 1280, CV_8UC3, fill);
    cv::Mat out;
    std::vector<int> target{1024, 1024};

    ASSERT_EQ(u.PreProcess(img, target, out), nullptr);
    ASSERT_EQ(out.size(), cv::Size(target[0], target[1]));
    ASSERT_EQ(out.channels(), 3);

    // Width drives resizing here (landscape). Width becomes 1024, height scales accordingly.
    int resized_w = target[0];
    int resized_h = static_cast<int>(img.rows / (img.cols / static_cast<float>(target[0])));

    // PreProcess converts BGR -> RGB, so expected color is swapped.
    cv::Scalar expected_rgb(fill[2], fill[1], fill[0]);

    // The top-left region (resized content) should keep the image color.
    cv::Mat roi_top = out(cv::Rect(0, 0, resized_w, resized_h));
    cv::Scalar mean_top = cv::mean(roi_top);
    EXPECT_NEAR(mean_top[0], expected_rgb[0], 1.0);
    EXPECT_NEAR(mean_top[1], expected_rgb[1], 1.0);
    EXPECT_NEAR(mean_top[2], expected_rgb[2], 1.0);

    // The area below the resized content (padding) must be zeros.
    if (resized_h < target[1]) {
        cv::Mat roi_pad = out(cv::Rect(0, resized_h, target[0], target[1] - resized_h));
        cv::Mat gray; cv::cvtColor(roi_pad, gray, cv::COLOR_BGR2GRAY);
        EXPECT_EQ(cv::countNonZero(gray), 0);
    }
}

// Explicitly ensure imgSize is interpreted as [W, H] in PreProcess for non-square targets.
TEST_F(UtilsTest, PreprocessNonSquareWidthHeightOrder) {
    // Input image: H=300, W=500
    cv::Mat img(300, 500, CV_8UC3, cv::Scalar(5, 6, 7));

    // Target canvas (W,H) with non-square dims
    std::vector<int> target{640, 480};
    cv::Mat out;

    ASSERT_EQ(u.PreProcess(img, target, out), nullptr);
    // cols = width, rows = height
    EXPECT_EQ(out.cols, target[0]);
    EXPECT_EQ(out.rows, target[1]);
    EXPECT_EQ(out.size(), cv::Size(target[0], target[1]));
}

// Parameterized fixture: used with TEST_P to run the same test body
// for many (input size, target size) pairs.
class UtilsPreprocessParamTest
    : public ::testing::TestWithParam<std::tuple<cv::Size, cv::Size>> {
protected:
    Utils u;
};

// TEST_P defines a parameterized test. It runs once per parameter set.
// We assert that:
// - Output size equals the target canvas.
// - Output has 3 channels (RGB).
// - The padding area (bottom or right) is zero depending on which side letterboxes.
TEST_P(UtilsPreprocessParamTest, LetterboxWithinBoundsAndChannels3) {
    const auto [inSize, target] = GetParam();
    cv::Mat img(inSize, CV_8UC3, cv::Scalar(1, 2, 3));
    cv::Mat out;

    ASSERT_EQ(u.PreProcess(img, {target.width, target.height}, out), nullptr);
    EXPECT_EQ(out.size(), target);
    EXPECT_EQ(out.channels(), 3);

    // Detect which side letterboxes and check that the padded region is zeros.
    if (inSize.width >= inSize.height) {
        int resized_h = static_cast<int>(inSize.height / (inSize.width / static_cast<float>(target.width)));
        if (resized_h < target.height) {
            cv::Mat roi_pad = out(cv::Rect(0, resized_h, target.width, target.height - resized_h));
            cv::Mat gray; cv::cvtColor(roi_pad, gray, cv::COLOR_BGR2GRAY);
            EXPECT_EQ(cv::countNonZero(gray), 0);
        }
    } else {
        int resized_w = static_cast<int>(inSize.width / (inSize.height / static_cast<float>(target.height)));
        if (resized_w < target.width) {
            cv::Mat roi_pad = out(cv::Rect(resized_w, 0, target.width - resized_w, target.height));
            cv::Mat gray; cv::cvtColor(roi_pad, gray, cv::COLOR_BGR2GRAY);
            EXPECT_EQ(cv::countNonZero(gray), 0);
        }
    }
}

// INSTANTIATE_TEST_SUITE_P provides the concrete parameter values.
// Each pair (input size, target size) creates a separate test instance.
INSTANTIATE_TEST_SUITE_P(
    ManySizes,
    UtilsPreprocessParamTest,
    ::testing::Values(
        std::make_tuple(cv::Size(640, 640),  cv::Size(1024, 1024)), // square -> square
        std::make_tuple(cv::Size(800, 600),  cv::Size(800, 600)),    // same size (no resize)
        std::make_tuple(cv::Size(600, 800),  cv::Size(800, 600)),    // portrait -> landscape
        std::make_tuple(cv::Size(1280, 720), cv::Size(1024, 1024))   // wide -> square
    )
);

// Separate fixture for point scaling tests.
class UtilsScaleBboxPointsTest : public ::testing::Test {
protected:
    Utils u;
};

// If the input size and target size are the same, scaling should do nothing.
TEST_F(UtilsScaleBboxPointsTest, IdentityWhenSameSize) {
    cv::Mat img(600, 800, CV_8UC3);
    std::vector<int> target{800, 600};
    std::vector<float> pts{100.f, 100.f, 700.f, 500.f};
    std::vector<float> scaled;

    u.ScaleBboxPoints(img, target, pts, scaled);
    ASSERT_EQ(scaled.size(), pts.size());
    EXPECT_NEAR(scaled[0], pts[0], 1e-3);
    EXPECT_NEAR(scaled[1], pts[1], 1e-3);
    EXPECT_NEAR(scaled[2], pts[2], 1e-3);
    EXPECT_NEAR(scaled[3], pts[3], 1e-3);
}

// When width drives the resize (landscape), both x and y are scaled by the same factor.
// We expect coordinates to be multiplied by target_width / input_width.
TEST_F(UtilsScaleBboxPointsTest, ScalesWidthDominant) {
    cv::Mat img(300, 600, CV_8UC3);                  // h=300, w=600 (w >= h)
    std::vector<int> target{1200, 600};              // width doubles
    std::vector<float> pts{100.f, 50.f, 500.f, 250.f};
    std::vector<float> scaled;

    u.ScaleBboxPoints(img, target, pts, scaled);
    ASSERT_EQ(scaled.size(), pts.size());
    const float scale = target[0] / static_cast<float>(img.cols); // 1200/600 = 2
    EXPECT_NEAR(scaled[0], pts[0] * scale, 1e-3);
    EXPECT_NEAR(scaled[1], pts[1] * scale, 1e-3);
    EXPECT_NEAR(scaled[2], pts[2] * scale, 1e-3);
    EXPECT_NEAR(scaled[3], pts[3] * scale, 1e-3);
}

// When height drives the resize (portrait), both x and y are scaled by the same factor.
// We expect coordinates to be multiplied by target_height / input_height.
TEST_F(UtilsScaleBboxPointsTest, ScalesHeightDominant) {
    cv::Mat img(600, 300, CV_8UC3);                  // h=600, w=300 (h > w)
    std::vector<int> target{600, 1200};              // height doubles
    std::vector<float> pts{100.f, 50.f, 200.f, 500.f};
    std::vector<float> scaled;

    u.ScaleBboxPoints(img, target, pts, scaled);
    ASSERT_EQ(scaled.size(), pts.size());
    const float scale = target[1] / static_cast<float>(img.rows); // 1200/600 = 2
    EXPECT_NEAR(scaled[0], pts[0] * scale, 1e-3);
    EXPECT_NEAR(scaled[1], pts[1] * scale, 1e-3);
    EXPECT_NEAR(scaled[2], pts[2] * scale, 1e-3);
    EXPECT_NEAR(scaled[3], pts[3] * scale, 1e-3);
}
