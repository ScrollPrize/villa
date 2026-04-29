#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <sys/resource.h>

#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"

namespace fs = std::filesystem;

namespace {

bool isTifxyzDir(const fs::path& dir)
{
    return fs::is_directory(dir) &&
           fs::exists(dir / "meta.json") &&
           fs::exists(dir / "x.tif") &&
           fs::exists(dir / "y.tif") &&
           fs::exists(dir / "z.tif");
}

long maxRssKb()
{
    rusage usage {};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return 0;
    }
    return usage.ru_maxrss;
}

void usage(const char* argv0)
{
    std::cerr << "Usage: " << argv0 << " <tifxyz-parent-folder> [--stride N] [--limit N]\n";
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        usage(argv[0]);
        return 2;
    }

    fs::path root;
    int stride = 1;
    std::size_t limit = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--stride" && i + 1 < argc) {
            stride = std::max(1, std::stoi(argv[++i]));
        } else if (arg == "--limit" && i + 1 < argc) {
            limit = static_cast<std::size_t>(std::stoull(argv[++i]));
        } else if (root.empty()) {
            root = arg;
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (root.empty() || !fs::is_directory(root)) {
        usage(argv[0]);
        return 2;
    }

    std::vector<fs::path> paths;
    for (const auto& entry : fs::directory_iterator(root)) {
        if (entry.is_directory() && isTifxyzDir(entry.path())) {
            paths.push_back(entry.path());
        }
    }
    std::sort(paths.begin(), paths.end());
    if (limit > 0 && paths.size() > limit) {
        paths.resize(limit);
    }

    std::vector<SurfacePatchIndex::SurfacePtr> surfaces;
    surfaces.reserve(paths.size());
    for (const auto& path : paths) {
        surfaces.push_back(std::make_shared<QuadSurface>(path));
    }

    std::cout << "surfaces=" << surfaces.size()
              << " stride=" << stride
              << " rss_before_kb=" << maxRssKb()
              << std::endl;

    SurfacePatchIndex index;
    index.setSamplingStride(stride);
    const auto start = std::chrono::steady_clock::now();
    index.rebuild(surfaces);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    std::cout << "seconds=" << seconds
              << " indexed_surfaces=" << index.surfaceCount()
              << " patches=" << index.patchCount()
              << " rss_after_kb=" << maxRssKb()
              << std::endl;

    return 0;
}
