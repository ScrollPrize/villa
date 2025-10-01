#include "vc/core/util/Surface.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/ChunkedTensor.hpp"
#include <opencv2/imgcodecs.hpp>
#include <iostream>

namespace fs = std::filesystem;

int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cout << "usage: " << argv[0] << " <tiffxyz-segment> [zarr-volume] [output-mask-path] [--overwrite]" << std::endl;
        std::cout << "  Generates a mask (and optionally image layer if volume provided)" << std::endl;
        std::cout << "  --overwrite: overwrite existing mask file (defaults to false)" << std::endl;
        return EXIT_SUCCESS;
    }

    fs::path seg_path = argv[1];

    // Parse arguments
    fs::path volume_path;
    fs::path mask_path = seg_path / "mask.tif";
    bool overwrite = false;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--overwrite") {
            overwrite = true;
        } else if (volume_path.empty() && fs::exists(arg) && fs::is_directory(arg)) {
            volume_path = arg;
        } else if (!arg.starts_with("--")) {
            mask_path = arg;
        }
    }

    // Check if mask already exists
    if (fs::exists(mask_path) && !overwrite) {
        std::cout << "Mask already exists at " << mask_path << std::endl;
        std::cout << "Use --overwrite flag to regenerate" << std::endl;
        return EXIT_SUCCESS;
    }

    // Load the surface
    QuadSurface *surf = nullptr;
    try {
        surf = load_quad_from_tifxyz(seg_path);
    }
    catch (const std::exception& e) {
        std::cerr << "Error loading surface: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    cv::Mat_<uint8_t> mask;
    cv::Mat_<uint8_t> img;

    // If volume path provided, generate with image data
    if (!volume_path.empty()) {
        std::shared_ptr<Volume> volume;
        ChunkCache* cache = nullptr;

        try {
            volume = Volume::New(volume_path);
            cache = new ChunkCache(1ULL * 1024ULL * 1024ULL * 1024ULL);

            //generate_mask(surf, mask, img, volume->zarrDataset(0), cache);

            // Save as multi-layer TIFF
            std::vector<cv::Mat> layers = {mask, img};
            if (!cv::imwritemulti(mask_path.string(), layers)) {
                std::cerr << "Error writing mask to " << mask_path << std::endl;
                delete cache;
                delete surf;
                return EXIT_FAILURE;
            }

            delete cache;
        }
        catch (const std::exception& e) {
            std::cerr << "Error processing volume: " << e.what() << std::endl;
            if (cache) delete cache;
            delete surf;
            return EXIT_FAILURE;
        }
    }

    std::cout << "Mask generated successfully at " << mask_path << std::endl;
    std::cout << "  Dimensions: " << mask.size() << std::endl;

    // Report statistics
    int valid_count = cv::countNonZero(mask);
    int total_count = mask.rows * mask.cols;
    float valid_percent = (float)valid_count / total_count * 100.0f;
    std::cout << "  Valid pixels: " << valid_count << " / " << total_count
              << " (" << std::fixed << std::setprecision(1) << valid_percent << "%)" << std::endl;

    delete surf;
    return EXIT_SUCCESS;
}