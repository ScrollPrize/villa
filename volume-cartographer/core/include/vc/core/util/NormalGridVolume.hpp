#pragma once

#include <cstdint>
#include <string>
#include <memory>
#include <optional>
#include <filesystem>

#include <opencv2/core/types.hpp>
#include "utils/Json.hpp"

#include "vc/core/util/GridStore.hpp"

namespace vc::core::util {

    // Marker file inside a normal-grid directory that turns it into a streaming
    // cache: {"url": "<https prefix>"}. The directory must be the URL's place under
    // a cache root, <root>/<scheme>/<host>/<path> (vcr::zarr::decodedCachePath, as
    // VC3D lays it out); grid files missing locally are downloaded there by
    // vc_reader's file cache under that root's disk budget and free-space floor.
    inline constexpr const char* kNormalGridsRemoteMarker = "normal-grids-remote.json";

    // The URL a normal-grid directory's marker records (trailing slashes
    // trimmed); nullopt without a readable marker.
    [[nodiscard]] std::optional<std::string> normalGridsRemoteUrl(const std::filesystem::path& dir);
    // Make dir (created if needed) a streaming cache of `url`; a marker already
    // naming it is kept. False when the marker cannot be written.
    bool writeNormalGridsRemoteMarker(const std::filesystem::path& dir, const std::string& url);
    // dir holds a whole normal-grid store: metadata.json and the xy, xz and yz grids.
    [[nodiscard]] bool hasNormalGridFiles(const std::filesystem::path& dir);

    class NormalGridVolume {
    public:
        struct CacheStats {
            uint64_t gridHits = 0;
            uint64_t gridMisses = 0;
            size_t liveGridEntries = 0;
            uint64_t decodedPathHits = 0;
            uint64_t decodedPathMisses = 0;
            uint64_t decodedPathEvictions = 0;
            size_t decodedPathEntries = 0;
            size_t decodedPathBytes = 0;
            size_t bucketIndexEntries = 0;
            size_t bucketIndexBytes = 0;
            // Streaming stores: fetches that failed for a reason other than the
            // file not existing (network, low disk space), and the last such message.
            uint64_t fetchFailures = 0;
            std::string lastFetchError;
        };

        explicit NormalGridVolume(const std::string& path);
        NormalGridVolume(const std::string& path, int level);
        ~NormalGridVolume();
        NormalGridVolume(NormalGridVolume&&) noexcept;
        NormalGridVolume& operator=(NormalGridVolume&&) noexcept;

        struct GridQueryResult {
            std::shared_ptr<const GridStore> grid1;
            std::shared_ptr<const GridStore> grid2;
            double weight;
        };

        std::optional<GridQueryResult> query(const cv::Point3f& point, int plane_idx) const;
        std::shared_ptr<const GridStore> query_nearest(const cv::Point3f& point, int plane_idx) const;
        std::shared_ptr<const GridStore> get_grid(int plane_idx, int slice_idx) const;
        CacheStats cacheStats() const;
        void resetCacheStats() const;
        double coordinateScale() const;
        double outputSpiralStep() const;
        int level() const;

    public:
        const utils::Json& metadata() const;

    private:
        struct pimpl;
        std::unique_ptr<pimpl> pimpl_;
    };

} // namespace vc::core::util
