// vc_zarr_to_tiff: Read a zarr dataset and write a TIFF stack

#include <boost/program_options.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>

#include <nlohmann/json.hpp>

#include <opencv2/core.hpp>

#include "vc/core/types/VcDataset.hpp"
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

    auto ds = std::make_unique<vc::VcDataset>(inRoot / dsName);

    const auto& shape = ds->shape(); // [Z, Y, X]
    if (shape.size() != 3) {
        std::cerr << "Expected 3D dataset, got " << shape.size() << "D\n";
        return 1;
    }
    const size_t Z = shape[0], Y = shape[1], X = shape[2];
    const auto dtype = ds->getDtype();

    int cvType;
    if (dtype == vc::VcDtype::uint8)
        cvType = CV_8UC1;
    else if (dtype == vc::VcDtype::uint16)
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
        std::vector<size_t> offset = {z, 0, 0};
        std::vector<size_t> regionShape = {1, Y, X};
        ds->readRegion(offset, regionShape, slice.data);

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
