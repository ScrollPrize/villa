// vcr::files::publishPath: new and existing destinations, files and directories.
// Pass a directory on NFS or exFAT, which refuse atomic no-replace and exchange
// renames, to test the fallback there.
#include "vc_reader/publication.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <system_error>
#include <unistd.h>

namespace fs = std::filesystem;
using vcr::files::publishPath;

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition); \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

static void put(const fs::path& path, const std::string& text) { std::ofstream(path, std::ios::binary) << text; }

static std::string get(const fs::path& path)
{
    std::ifstream in(path, std::ios::binary);
    return in ? std::string(std::istreambuf_iterator<char>(in), {}) : "<missing>";
}

// The error a publication reports, empty when it succeeds.
template <class Publish>
static std::error_code failure(Publish&& publish)
{
    try {
        publish();
    } catch (const fs::filesystem_error& e) {
        return e.code();
    }
    return {};
}

int main(int argc, char** argv)
{
    const auto root = (argc > 1 ? fs::path(argv[1]) : fs::temp_directory_path()) /
                      ("vc_reader_publication_" + std::to_string(::getpid()));
    fs::create_directories(root);

    // A new file. Where the atomic rename is refused the file is hard linked, so a
    // filesystem without hard links either (exFAT) reports that and changes nothing.
    put(root / "probe", "");
    std::error_code noLinks, ignored;
    fs::create_hard_link(root / "probe", root / "probe.link", noLinks);
    fs::remove(root / "probe.link", ignored);
    fs::remove(root / "probe");
    put(root / "file.tmp", "new");
    if (const auto code = failure([&] { publishPath(root / "file.tmp", root / "file", false); })) {
        CHECK(noLinks && code == noLinks && !fs::exists(root / "file") && get(root / "file.tmp") == "new");
        fs::rename(root / "file.tmp", root / "file");
    }
    CHECK(get(root / "file") == "new" && !fs::exists(root / "file.tmp"));

    fs::create_directory(root / "dir.tmp");
    put(root / "dir.tmp" / "a", "new");
    publishPath(root / "dir.tmp", root / "dir", false);
    CHECK(get(root / "dir" / "a") == "new" && !fs::exists(root / "dir.tmp"));

    // Without overwrite an existing destination stays, and so does the source.
    put(root / "file.tmp", "newer");
    CHECK(failure([&] { publishPath(root / "file.tmp", root / "file", false); }) == std::errc::file_exists);
    CHECK(get(root / "file") == "new" && get(root / "file.tmp") == "newer");

    // Overwrite exchanges, leaving the old destination at the source.
    publishPath(root / "file.tmp", root / "file", true);
    CHECK(get(root / "file") == "newer" && get(root / "file.tmp") == "new");
    fs::create_directory(root / "dir.tmp");
    put(root / "dir.tmp" / "a", "newer");
    publishPath(root / "dir.tmp", root / "dir", true);
    CHECK(get(root / "dir" / "a") == "newer" && get(root / "dir.tmp" / "a") == "new");

    // An exchange that cannot complete keeps the destination: a directory cannot
    // be exchanged with its own parent.
    fs::create_directory(root / "dir" / "inner");
    CHECK(failure([&] { publishPath(root / "dir" / "inner", root / "dir", true); }));
    CHECK(get(root / "dir" / "a") == "newer" && fs::is_directory(root / "dir" / "inner"));

    // Nothing is left beside the four names.
    CHECK(std::distance(fs::directory_iterator(root), fs::directory_iterator()) == 4);
    fs::remove_all(root);
    std::puts("publication checks passed");
    return 0;
}
