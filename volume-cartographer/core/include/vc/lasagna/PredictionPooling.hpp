#pragma once
#include <vc_reader/zarr.hpp>

#include <filesystem>
#include <memory>

namespace vc::lasagna {
// The pooled view of a local fiber prediction store: a directory holding the
// view's .zarray and .native_pool.json (recipe, source store, channel). It is a
// vcr::computed array whose chunks are pooled 2x on demand from the source
// chunks, which are read through vc_reader with `cache` as their memory cache.
vcr::zarr::OpenedArray openPooledPredictionArray(
    const std::filesystem::path& path, std::shared_ptr<vcr::MemoryCache> cache);
}
