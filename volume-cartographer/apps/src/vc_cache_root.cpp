#include "vc/core/util/GlobalCacheRoot.hpp"
#include "vc/core/util/AtomicFile.hpp"
#include <chrono>
#include <iostream>

int main(int argc, char** argv)
{
    try {
        if (argc == 1) {
            std::cout << vc::core::util::globalCacheRoot().string() << '\n';
            return 0;
        }
        if (argc != 3 || std::string_view(argv[1]) != "--set")
            throw std::invalid_argument("usage: vc_cache_root [--set /absolute/cache/root]");
        const std::filesystem::path root(argv[2]);
        if (!root.is_absolute()) throw std::invalid_argument("cache root must be absolute");
        std::filesystem::create_directories(root);
        // An existing directory need not be writable. Test before publishing a
        // configuration that would make subsequent GUI/headless startup fail.
        const auto probe = root / (".vc-cache-write-probe-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
        if (!std::filesystem::create_directory(probe))
            throw std::runtime_error("cannot create cache write probe");
        try {
            vc::core::util::atomicWriteString(probe / "probe", "write test\n");
            std::filesystem::remove(probe / "probe");
            std::filesystem::remove(probe);
        } catch (...) {
            std::error_code ignored;
            std::filesystem::remove(probe / "probe", ignored);
            std::filesystem::remove(probe, ignored);
            throw;
        }
        const auto directory = vc::core::util::vc3dConfigDirectory();
        std::filesystem::create_directories(directory);
        const auto path = directory / "cache.json";
        if (std::filesystem::exists(path)) {
            const auto suffix = std::chrono::system_clock::now().time_since_epoch().count();
            std::filesystem::copy_file(path, directory / ("cache.json.backup-" + std::to_string(suffix)));
        }
        vc::core::util::atomicWriteString(path, nlohmann::json{
            {"version", 1}, {"root", root.lexically_normal().string()}}.dump(2) + "\n");
        std::cout << vc::core::util::globalCacheRoot().string() << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
