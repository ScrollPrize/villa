#include "VolumePreview.hpp"
#include "fibers3d/Fiber3DCT.hpp"
#include "vc/core/types/Volume.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vc3d::scene3d
{
std::shared_ptr<VolumeData> sampleVolume(Volume& volume, int requestedLevel, size_t maxBytes,
                                       const std::atomic_bool& cancelled)
{
    auto checkCancelled = [&] {
        if (cancelled.load()) throw std::runtime_error("CT volume loading cancelled");
    };
    checkCancelled();
    if (maxBytes < 128) throw std::invalid_argument("CT preview needs at least 128 bytes");
    const auto native = volume.shapeXyz();
    if (std::any_of(native.begin(), native.end(), [](int n) { return n <= 0; }))
        throw std::runtime_error("Invalid CT volume dimensions");
    auto levels = volume.presentScaleLevels();
    std::sort(levels.begin(), levels.end());
    if (levels.empty()) throw std::runtime_error("CT volume has no available Zarr levels");
    constexpr int maxEdge = 256; // guaranteed minimum 3D texture size in OpenGL 3.3
    const size_t scratchBytes = std::min(maxBytes/4, size_t(256*1024));
    const size_t textureBytes = std::min(maxBytes-scratchBytes, size_t(64*1024*1024));
    const size_t maxVoxels = textureBytes/4; // RG8 on CPU and GPU
    const int batchSize = int(std::max(size_t(1), scratchBytes/64));
    auto fits = [&](const std::array<int,3>& shape) {
        size_t count = 1;
        for (int n : shape) {
            if (n <= 0 || n > maxEdge || count > maxVoxels/size_t(n)) return false;
            count *= size_t(n);
        }
        return true;
    };
    auto selected = std::lower_bound(levels.begin(), levels.end(), std::max(0, requestedLevel));
    if (selected == levels.end()) selected = std::prev(levels.end());
    while (std::next(selected) != levels.end() && !fits(volume.levelShape(*selected))) ++selected;
    const int level = *selected;
    const auto sourceZYX = volume.levelShape(level);
    if (std::any_of(sourceZYX.begin(), sourceZYX.end(), [](int n) { return n <= 0; }))
        throw std::runtime_error("Invalid CT pyramid dimensions");
    std::array<int,3> shape{sourceZYX[2],sourceZYX[1],sourceZYX[0]};
    if (!fits(shape)) {
        // One common reduction factor keeps the native aspect ratio. A binary
        // search handles tiny budgets without overflow or a full-sized buffer.
        double lo = 1, hi = *std::max_element(shape.begin(),shape.end());
        const auto source = shape;
        auto reduced = [&](double factor) {
            std::array<int,3> result;
            for (int k=0; k<3; ++k) result[k]=std::max(1,int(std::ceil(source[k]/factor)));
            return result;
        };
        for (int i=0; i<64; ++i) {
            const double middle = (lo+hi)*.5;
            if (fits(reduced(middle))) hi=middle;
            else lo=middle;
        }
        shape = reduced(hi);
    }
    auto data = std::make_shared<VolumeData>();
    data->requestedLevel = requestedLevel;
    data->resampled = shape != std::array<int,3>{sourceZYX[2],sourceZYX[1],sourceZYX[0]};
    for (int k=0; k<3; ++k) data->bounds.upper[k]=native[k]-1;
    auto image = std::make_shared<fibers3d::slices::Image>();
    image->level = level;
    auto& slab = image->slab.emplace();
    slab.width=shape[0]; slab.height=shape[1]; slab.depth=shape[2];
    // Preserve the native voxel faces, independent of Zarr level or preview
    // resolution. Resampling changes detail, never physical scale/registration.
    slab.origin={-.5,-.5,-.5};
    slab.right={double(native[0]),0,0}; slab.down={0,double(native[1]),0}; slab.through={0,0,double(native[2])};
    const size_t voxels=size_t(shape[0])*shape[1]*shape[2];
    slab.pixels.resize(2*voxels);
    const double levelStep=std::ldexp(1.,level);
    if (!std::isfinite(levelStep)) throw std::runtime_error("Invalid CT pyramid level");
    bool visible=false;
    for (size_t first=0; first<voxels; first+=batchSize) {
        checkCancelled();
        const int count=int(std::min(voxels-first,size_t(batchSize)));
        cv::Mat_<cv::Vec3f> coords(1,count);
        for (int i=0; i<count; ++i) {
            const size_t index=first+i;
            const int cell[3]={int(index%shape[0]),int((index/shape[0])%shape[1]),int(index/(size_t(shape[0])*shape[1]))};
            for (int k=0; k<3; ++k) {
                const double center=-.5+(cell[k]+.5)*double(native[k])/shape[k];
                // Hold the edge sample outside a coarse level's last sample
                // center, rather than introducing a dark band at the boundary.
                const double last=std::min(double(native[k]-1),(sourceZYX[2-k]-1)*levelStep);
                coords(0,i)[k]=float(std::clamp(center,0.,last));
            }
        }
        const auto samples=fibers3d::ct::sampleCoordinates(coords,volume,level);
        for (int i=0; i<count; ++i) {
            slab.pixels[2*(first+i)]=samples(0,i)[0];
            slab.pixels[2*(first+i)+1]=samples(0,i)[1];
            visible |= samples(0,i)[0] != 0 && samples(0,i)[1] != 0;
        }
    }
    if (!visible) return {};
    data->images.push_back(std::move(image));
    return data;
}
}
