#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/Slicing.hpp"


void render_binary_mask(QuadSurface* surf,
                         cv::Mat_<uint8_t>& mask,
                         cv::Mat_<cv::Vec3f>& coords_out) {

    cv::Size nominalSize = surf->size();
    cv::Size outputSize = nominalSize;

    cv::Vec3f ptr = surf->pointer();
    cv::Vec3f offset(-nominalSize.width/2.0f, -nominalSize.height/2.0f, 0);

    surf->gen(&coords_out, nullptr, outputSize, ptr, 1.0f, offset);

    mask.create(outputSize);
#pragma omp parallel for schedule(dynamic, 1)
    for(int j = 0; j < outputSize.height; j++) {
        for(int i = 0; i < outputSize.width; i++) {
            const cv::Vec3f& coord = coords_out(j, i);
            mask(j, i) = (std::isfinite(coord[0]) &&
                         std::isfinite(coord[1]) &&
                         std::isfinite(coord[2])) ? 255 : 0;
        }
    }

    std::cout << "render_binary_mask: output size=" << outputSize << std::endl;
}

void render_image_from_coords(const cv::Mat_<cv::Vec3f>& coords,
                              cv::Mat_<uint8_t>& img,
                              z5::Dataset* ds,
                              ChunkCache* cache) {
    if (!ds || !cache) {
        throw std::runtime_error("Dataset or cache is null in render_image_from_coords");
    }

    readInterpolated3D(img, ds, coords, cache);
    std::cout << "render_image_from_coords: completed" << std::endl;
}

// Render surface - generates both mask and image
void render_surface_image(QuadSurface* surf,
                         cv::Mat_<uint8_t>& mask,
                         cv::Mat_<uint8_t>& img,
                         z5::Dataset* ds,
                         ChunkCache* cache) {

    cv::Mat_<cv::Vec3f> coords;
    render_binary_mask(surf, mask, coords);
    render_image_from_coords(coords, img, ds, cache);

    std::cout << "render_surface_image: completed" << std::endl;
}