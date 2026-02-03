// vc_zarr_to_tiff: Read a zarr dataset and write a TIFF stack

#include <boost/program_options.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>

#include <nlohmann/json.hpp>
#include "z5/factory.hxx"
#include "z5/filesystem/handle.hxx"
#include "z5/filesystem/dataset.hxx"
#include "z5/multiarray/xtensor_access.hxx"
#include <xtensor/containers/xarray.hpp>

#include <opencv2/core.hpp>
#include <tiffio.h>
#include "vc/core/util/Tiff.hpp"

namespace po = boost::program_options;
namespace fs = std::filesystem;
using json = nlohmann::json;

static uint16_t parseCompression(const std::string& s)
{
    if (s == "packbits") return COMPRESSION_PACKBITS;
    if (s == "lzw") return COMPRESSION_LZW;
    if (s == "deflate") return COMPRESSION_DEFLATE;
    if (s == "none") return COMPRESSION_NONE;
    throw std::runtime_error("Unknown compression: " + s);
}

int main(int argc, char** argv)
{
    std::string inputPath, outputPath, compressionStr;
    int level = 0;

    po::options_description desc("vc_zarr_to_tiff options");
    desc.add_options()
        ("help,h", "Show help")
        ("input,i", po::value<std::string>(&inputPath)->required(),
         "Input zarr directory")
        ("output,o", po::value<std::string>(&outputPath)->required(),
         "Output directory for TIFF files")
        ("level,l", po::value<int>(&level)->default_value(0),
         "Pyramid level to read (default 0)")
        ("compression,c", po::value<std::string>(&compressionStr)->default_value("packbits"),
         "TIFF compression: packbits, lzw, deflate, none");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << desc << "\n";
            return 0;
        }
        po::notify(vm);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n" << desc << "\n";
        return 1;
    }

    const uint16_t compression = parseCompression(compressionStr);

    // Open zarr dataset
    fs::path inRoot(inputPath);
    std::string dsName = std::to_string(level);
    std::string dimsep = ".";
    try {
        json j = json::parse(
            std::ifstream((inRoot / dsName / ".zarray").string()));
        if (j.contains("dimension_separator"))
            dimsep = j["dimension_separator"].get<std::string>();
    } catch (...) {}

    z5::filesystem::handle::Group root(inRoot, z5::FileMode::FileMode::r);
    z5::filesystem::handle::Dataset dsHandle(root, dsName, dimsep);
    auto ds = z5::filesystem::openDataset(dsHandle);

    const auto& shape = ds->shape(); // [Z, Y, X]
    if (shape.size() != 3) {
        std::cerr << "Expected 3D dataset, got " << shape.size() << "D\n";
        return 1;
    }
    const size_t Z = shape[0], Y = shape[1], X = shape[2];
    const auto dtype = ds->getDtype();

    int cvType;
    if (dtype == z5::types::Datatype::uint8)
        cvType = CV_8UC1;
    else if (dtype == z5::types::Datatype::uint16)
        cvType = CV_16UC1;
    else {
        std::cerr << "Unsupported dtype (need uint8 or uint16)\n";
        return 1;
    }

    std::cout << "Dataset: " << X << "x" << Y << "x" << Z
              << " dtype=" << (cvType == CV_8UC1 ? "uint8" : "uint16")
              << " level=" << level << "\n";

    // Create output directory
    fs::path outDir(outputPath);
    fs::create_directories(outDir);

    // Determine filename width
    int numWidth = std::max(2, static_cast<int>(std::to_string(Z - 1).size()));

    // Process each Z slice
    for (size_t z = 0; z < Z; ++z) {
        std::cout << "\r[" << (z + 1) << "/" << Z << "]" << std::flush;

        cv::Mat slice(static_cast<int>(Y), static_cast<int>(X), cvType);
        z5::types::ShapeType offset = {z, 0, 0};

        if (cvType == CV_8UC1) {
            xt::xarray<uint8_t> slab = xt::empty<uint8_t>({1ul, Y, X});
            z5::multiarray::readSubarray<uint8_t>(ds, slab, offset.begin());
            for (size_t y = 0; y < Y; ++y) {
                auto* row = slice.ptr<uint8_t>(static_cast<int>(y));
                for (size_t x = 0; x < X; ++x) row[x] = slab(0, y, x);
            }
        } else {
            xt::xarray<uint16_t> slab = xt::empty<uint16_t>({1ul, Y, X});
            z5::multiarray::readSubarray<uint16_t>(ds, slab, offset.begin());
            for (size_t y = 0; y < Y; ++y) {
                auto* row = slice.ptr<uint16_t>(static_cast<int>(y));
                for (size_t x = 0; x < X; ++x) row[x] = slab(0, y, x);
            }
        }

        // Write TIFF
        std::ostringstream fname;
        fname << std::setw(numWidth) << std::setfill('0') << z << ".tif";
        fs::path outPath = outDir / fname.str();

        constexpr uint32_t tileSize = 256;
        TiffWriter writer(outPath,
                          static_cast<uint32_t>(X),
                          static_cast<uint32_t>(Y),
                          cvType, tileSize, tileSize,
                          0.0f, compression);

        for (uint32_t ty = 0; ty < Y; ty += tileSize) {
            for (uint32_t tx = 0; tx < X; tx += tileSize) {
                uint32_t tw = std::min(tileSize, static_cast<uint32_t>(X) - tx);
                uint32_t th = std::min(tileSize, static_cast<uint32_t>(Y) - ty);
                cv::Mat tile = slice(
                    cv::Range(static_cast<int>(ty), static_cast<int>(ty + th)),
                    cv::Range(static_cast<int>(tx), static_cast<int>(tx + tw)));
                writer.writeTile(tx, ty, tile);
            }
        }
        writer.close();
    }

    std::cout << "\nDone. Wrote " << Z << " TIFFs to " << outDir << "\n";
    return 0;
}
