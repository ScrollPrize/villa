#include "test.hpp"

#include <opencv2/core.hpp>

#include <filesystem>

#include "vc/core/util/QuadSurface.hpp"

namespace fs = std::filesystem;

TEST(QuadSurfaceResample, SupportsAnisotropicFactors)
{
    cv::Mat_<cv::Vec3f> points(4, 4);
    for (int row = 0; row < points.rows; ++row) {
        for (int col = 0; col < points.cols; ++col) {
            points(row, col) = cv::Vec3f(static_cast<float>(col),
                                         static_cast<float>(row),
                                         1.0f);
        }
    }

    QuadSurface surf(points, cv::Vec2f(1.0f, 1.0f));
    surf.resample(2.0f, 0.5f, 1);

    EXPECT_EQ(surf.rawPointsPtr()->cols, 8);
    EXPECT_EQ(surf.rawPointsPtr()->rows, 2);
    EXPECT_FLOAT_EQ(surf._scale[0], 0.5f);
    EXPECT_FLOAT_EQ(surf._scale[1], 2.0f);
}

TEST(QuadSurfaceResample, UniformOverloadDelegatesToAxisWisePath)
{
    cv::Mat_<cv::Vec3f> points(3, 3, cv::Vec3f(1.0f, 2.0f, 3.0f));
    QuadSurface surf(points, cv::Vec2f(2.0f, 4.0f));

    surf.resample(2.0f, 1);

    EXPECT_EQ(surf.rawPointsPtr()->cols, 6);
    EXPECT_EQ(surf.rawPointsPtr()->rows, 6);
    EXPECT_FLOAT_EQ(surf._scale[0], 1.0f);
    EXPECT_FLOAT_EQ(surf._scale[1], 2.0f);
}

TEST(QuadSurfaceMeta, SaveMetaWritesRequiredTifxyzFields)
{
    cv::Mat_<cv::Vec3f> points(2, 2);
    points(0, 0) = cv::Vec3f(1.0f, 2.0f, 3.0f);
    points(0, 1) = cv::Vec3f(4.0f, 5.0f, 6.0f);
    points(1, 0) = cv::Vec3f(-1.0f, -1.0f, -1.0f);
    points(1, 1) = cv::Vec3f(7.0f, 8.0f, 9.0f);

    const fs::path dir = fs::temp_directory_path() / "vc_quad_surface_save_meta_required_fields";
    fs::remove_all(dir);
    fs::create_directories(dir);

    QuadSurface surf(points, cv::Vec2f(0.5f, 2.0f));
    surf.path = dir;
    surf.id = "surface-id";
    surf.meta = utils::Json::object();
    surf.meta["custom"] = "kept";

    surf.save_meta();

    const auto meta = utils::Json::parse_file(dir / "meta.json");
    EXPECT_EQ(meta["type"].get_string(), std::string("seg"));
    EXPECT_EQ(meta["uuid"].get_string(), std::string("surface-id"));
    EXPECT_EQ(meta["format"].get_string(), std::string("tifxyz"));
    EXPECT_FLOAT_EQ(meta["scale"][0].get_float(), 0.5f);
    EXPECT_FLOAT_EQ(meta["scale"][1].get_float(), 2.0f);
    EXPECT_TRUE(meta.contains("bbox"));
    EXPECT_EQ(meta["custom"].get_string(), std::string("kept"));

    fs::remove_all(dir);
}
