#include "test.hpp"

#include <sstream>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "vc/core/util/Thinning.hpp"

namespace {

std::string tracesSignature(const std::vector<std::vector<cv::Point>>& traces)
{
    std::stringstream out;
    for (const auto& trace : traces) {
        for (const auto& point : trace) {
            out << point.x << "," << point.y << ";";
        }
        out << "|";
    }
    return out.str();
}

cv::Mat makeStraightLine()
{
    cv::Mat image = cv::Mat::zeros(32, 32, CV_8U);
    cv::line(image, cv::Point(4, 16), cv::Point(27, 16), cv::Scalar(255), 3);
    return image;
}

cv::Mat makeTJunction()
{
    cv::Mat image = cv::Mat::zeros(32, 32, CV_8U);
    cv::line(image, cv::Point(16, 4), cv::Point(16, 27), cv::Scalar(255), 3);
    cv::line(image, cv::Point(8, 10), cv::Point(24, 10), cv::Scalar(255), 3);
    return image;
}

cv::Mat makeLoop()
{
    cv::Mat image = cv::Mat::zeros(40, 40, CV_8U);
    cv::circle(image, cv::Point(20, 20), 10, cv::Scalar(255), 3);
    return image;
}

cv::Mat makeFilledRectangle()
{
    cv::Mat image = cv::Mat::zeros(40, 40, CV_8U);
    cv::rectangle(image, cv::Rect(8, 10, 20, 16), cv::Scalar(255), cv::FILLED);
    return image;
}

void expectEquivalent(const cv::Mat& image)
{
    std::vector<std::vector<cv::Point>> tracesWithOutput;
    std::vector<std::vector<cv::Point>> tracesTraceOnly;
    cv::Mat outputOnly;
    cv::Mat outputWithTraces;

    customThinning(image, outputOnly, nullptr);
    customThinning(image, outputWithTraces, &tracesWithOutput);
    customThinningTraceOnly(image, tracesTraceOnly);

    EXPECT_EQ(tracesSignature(tracesWithOutput), tracesSignature(tracesTraceOnly));
    EXPECT_EQ(cv::countNonZero(outputOnly != outputWithTraces), 0);
}

} // namespace

TEST(ThinningEquivalence, StraightLine)
{
    expectEquivalent(makeStraightLine());
}

TEST(ThinningEquivalence, TJunction)
{
    expectEquivalent(makeTJunction());
}

TEST(ThinningEquivalence, ClosedLoop)
{
    expectEquivalent(makeLoop());
}

TEST(ThinningEquivalence, FilledRectangle)
{
    expectEquivalent(makeFilledRectangle());
}
