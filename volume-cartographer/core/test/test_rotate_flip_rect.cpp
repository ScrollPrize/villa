// Exhaustively checks that destination tile and band rectangles invert the shared transform.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <cstdint>
#include <opencv2/core.hpp>
#include "vc/core/util/RotateFlip.hpp"

TEST_CASE("destination rectangles invert rotate and flip")
{
    for (cv::Size size : {cv::Size(1,1), cv::Size(7,5), cv::Size(300,200), cv::Size(129,17), cv::Size(255,257)}) {
        cv::Mat src(size, CV_8UC1);
        for (int y = 0; y < size.height; ++y)
            for (int x = 0; x < size.width; ++x) src.at<uint8_t>(y,x) = uint8_t((x * 17 + y * 31) & 255);
        for (int rot : {-1, 0, 1, 2, 3}) for (int flip : {-1, 0, 1, 2}) {
            cv::Mat expected = src.clone();
            vc::rotateFlipIfNeeded(expected, rot, flip);
            for (cv::Size tile : {cv::Size(5, 3), cv::Size(16, 16), cv::Size(128, 128)})
                for (int y = 0; y < expected.rows; y += tile.height)
                    for (int x = 0; x < expected.cols; x += tile.width) {
                        cv::Rect dst(x, y, std::min(tile.width, expected.cols-x), std::min(tile.height, expected.rows-y));
                        cv::Mat got = src(vc::srcRectForDst(dst, size, rot, flip)).clone();
                        vc::rotateFlipIfNeeded(got, rot, flip);
                        CHECK(cv::countNonZero(got != expected(dst)) == 0);
                    }
            for (int band : {16, 128}) for (int y = 0; y < expected.rows; y += band) {
                cv::Rect dst(0, y, expected.cols, std::min(band, expected.rows-y));
                cv::Mat got = src(vc::srcRectForDst(dst, size, rot, flip)).clone();
                vc::rotateFlipIfNeeded(got, rot, flip);
                CHECK(cv::countNonZero(got != expected(dst)) == 0);
            }
        }
    }
}
