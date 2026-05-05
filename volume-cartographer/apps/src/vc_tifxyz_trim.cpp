#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/Tiff.hpp"

#include "utils/Json.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <tiffio.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;
using Json = utils::Json;

static void usage(const char* prog)
{
    std::cerr
        << "usage: " << prog << " <src tifxyz dir> [<dst tifxyz dir>]\n"
        << "  Crops x.tif/y.tif/z.tif and any sibling .tif channels (mask, etc.)\n"
        << "  to the bounding box of valid grid cells. Channels at integer multiples\n"
        << "  of the grid resolution are cropped in lockstep; channels with\n"
        << "  non-integer ratios are skipped.\n"
        << "  When <dst> is omitted or equal to <src>, the segment is replaced\n"
        << "  via a sibling temp dir + atomic rename.\n";
}

static bool writeChannelTiff(const fs::path& outPath, const cv::Mat& mat)
{
    if (mat.channels() == 1 &&
        (mat.type() == CV_8UC1 || mat.type() == CV_16UC1 || mat.type() == CV_32FC1))
    {
        try {
            writeTiff(outPath, mat, -1, 0, 0, -1.0f, COMPRESSION_NONE);
            return true;
        } catch (...) {
            // fall through to OpenCV
        }
    }
    std::vector<int> params = { cv::IMWRITE_TIFF_COMPRESSION, COMPRESSION_NONE };
    return cv::imwrite(outPath.string(), mat, params);
}

int main(int argc, char* argv[])
{
    if (argc != 2 && argc != 3) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    fs::path src = fs::absolute(fs::path(argv[1])).lexically_normal();
    fs::path dst = (argc == 3)
        ? fs::absolute(fs::path(argv[2])).lexically_normal()
        : src;

    if (!fs::is_directory(src)) {
        std::cerr << "error: source is not a directory: " << src << "\n";
        return EXIT_FAILURE;
    }

    std::unique_ptr<QuadSurface> surf;
    try {
        surf = load_quad_from_tifxyz(src);
    } catch (const std::exception& e) {
        std::cerr << "error loading " << src << ": " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    cv::Mat_<cv::Vec3f>* points = surf->rawPointsPtr();
    if (!points || points->empty()) {
        std::cerr << "error: empty points grid\n";
        return EXIT_FAILURE;
    }

    const int H = points->rows;
    const int W = points->cols;

    int r0 = H, r1 = -1, c0 = W, c1 = -1;
    for (int r = 0; r < H; ++r) {
        const cv::Vec3f* row = (*points)[r];
        int rc0 = W, rc1 = -1;
        for (int c = 0; c < W; ++c) {
            if (row[c][0] != -1.f) {
                if (c < rc0) rc0 = c;
                rc1 = c;
            }
        }
        if (rc1 >= 0) {
            if (r < r0) r0 = r;
            r1 = r;
            if (rc0 < c0) c0 = rc0;
            if (rc1 > c1) c1 = rc1;
        }
    }

    if (r1 < 0 || c1 < 0) {
        std::cerr << "error: no valid cells in grid, refusing to trim\n";
        return EXIT_FAILURE;
    }

    const int bbH = r1 - r0 + 1;
    const int bbW = c1 - c0 + 1;
    if (bbH == H && bbW == W) {
        std::cout << "already minimal (" << W << "x" << H << "), nothing to do\n";
        return EXIT_SUCCESS;
    }

    std::cout << "trim grid " << W << "x" << H << " -> " << bbW << "x" << bbH
              << " (rect c=" << c0 << "+" << bbW << " r=" << r0 << "+" << bbH << ")\n";

    fs::path stage = dst.parent_path() /
        (".vctifxyz_trim_" + std::to_string(::getpid()) + "_" + dst.filename().string());
    std::error_code ec;
    fs::remove_all(stage, ec);
    if (!fs::create_directories(stage, ec)) {
        std::cerr << "error: cannot create staging dir " << stage
                  << ": " << ec.message() << "\n";
        return EXIT_FAILURE;
    }

    const cv::Rect grid_rect(c0, r0, bbW, bbH);
    const cv::Mat_<cv::Vec3f> trimmed = (*points)(grid_rect).clone();
    std::vector<cv::Mat> xyz;
    cv::split(trimmed, xyz);
    writeTiff(stage / "x.tif", xyz[0], -1, 0, 0, -1.0f, COMPRESSION_NONE);
    writeTiff(stage / "y.tif", xyz[1], -1, 0, 0, -1.0f, COMPRESSION_NONE);
    writeTiff(stage / "z.tif", xyz[2], -1, 0, 0, -1.0f, COMPRESSION_NONE);

    for (const auto& entry : fs::directory_iterator(src)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() != ".tif") continue;
        const std::string stem = entry.path().stem().string();
        if (stem == "x" || stem == "y" || stem == "z") continue;

        std::vector<cv::Mat> layers;
        cv::imreadmulti(entry.path().string(), layers, cv::IMREAD_UNCHANGED);
        if (layers.empty() || layers[0].empty()) {
            std::cerr << "warning: failed to read " << entry.path().filename().string()
                      << ", skipping\n";
            continue;
        }
        const cv::Mat raw = layers[0];

        if (raw.cols < W || raw.rows < H ||
            (raw.cols % W) != 0 || (raw.rows % H) != 0) {
            std::cerr << "warning: " << entry.path().filename().string()
                      << " size " << raw.cols << "x" << raw.rows
                      << " is not an integer multiple of grid " << W << "x" << H
                      << ", skipping\n";
            continue;
        }
        const int sx = raw.cols / W;
        const int sy = raw.rows / H;
        const cv::Rect ch_rect(c0 * sx, r0 * sy, bbW * sx, bbH * sy);
        const cv::Mat ch_trim = raw(ch_rect).clone();

        if (!writeChannelTiff(stage / (stem + ".tif"), ch_trim)) {
            std::cerr << "error: failed to write " << (stem + ".tif") << "\n";
            return EXIT_FAILURE;
        }
        std::cout << "trim " << (stem + ".tif") << " "
                  << raw.cols << "x" << raw.rows << " -> "
                  << ch_trim.cols << "x" << ch_trim.rows << "\n";
    }

    Json meta = Json::parse_file(src / "meta.json");
    if (meta.contains("bbox")) meta.erase("bbox");
    {
        std::ofstream out(stage / "meta.json");
        out << meta.dump(4) << std::endl;
    }

    for (const auto& entry : fs::directory_iterator(src)) {
        if (!entry.is_regular_file()) continue;
        const fs::path& p = entry.path();
        if (p.extension() == ".tif") continue;
        if (p.filename() == "meta.json") continue;
        fs::copy_file(p, stage / p.filename(),
                      fs::copy_options::overwrite_existing, ec);
        if (ec) {
            std::cerr << "warning: failed to copy " << p.filename().string()
                      << ": " << ec.message() << "\n";
            ec.clear();
        }
    }

    if (fs::exists(dst)) {
        fs::path backup = dst.parent_path() /
            (".vctifxyz_trim_old_" + std::to_string(::getpid()) + "_" + dst.filename().string());
        fs::rename(dst, backup, ec);
        if (ec) {
            std::cerr << "error: cannot move existing " << dst
                      << " out of the way: " << ec.message() << "\n";
            fs::remove_all(stage);
            return EXIT_FAILURE;
        }
        fs::rename(stage, dst, ec);
        if (ec) {
            std::cerr << "error: cannot rename " << stage << " -> " << dst
                      << ": " << ec.message() << "\n";
            fs::rename(backup, dst);
            return EXIT_FAILURE;
        }
        fs::remove_all(backup);
    } else {
        fs::create_directories(dst.parent_path(), ec);
        fs::rename(stage, dst, ec);
        if (ec) {
            std::cerr << "error: cannot rename " << stage << " -> " << dst
                      << ": " << ec.message() << "\n";
            return EXIT_FAILURE;
        }
    }

    std::cout << "wrote trimmed segment to " << dst << "\n";
    return EXIT_SUCCESS;
}
