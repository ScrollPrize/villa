// vcr::files publication: publishPath stays strict, publishDirectory with overwrite
// completes what an interrupted save left, and the file by file publisher used where
// the no-replace rename is refused. Pass a directory on NFS or exFAT to run it there.
#include "vc_reader/publication.hpp"
#include "publication_files.hpp"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;
using vcr::files::publishDirectory;
using vcr::files::publishPath;
using vcr::files::detail::publishNewByFiles;
using vcr::files::detail::renameFlagRefused;

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition); \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

static const std::vector<std::string> kNames{"meta.json", "x.tif", "y.tif", "z.tif"};

static void put(const fs::path& path, const std::string& text) { std::ofstream(path, std::ios::binary) << text; }

static std::string get(const fs::path& path)
{
    std::ifstream in(path, std::ios::binary);
    return in ? std::string(std::istreambuf_iterator<char>(in), {}) : "<missing>";
}

// A staged segment: every file holds its name after the tag.
static void stage(const fs::path& dir, const std::string& tag)
{
    fs::create_directories(dir);
    for (const auto& name : kNames) put(dir / name, tag + name);
}

static bool holds(const fs::path& dir, const std::string& tag)
{
    for (const auto& name : kNames)
        if (get(dir / name) != tag + name) return false;
    return std::distance(fs::directory_iterator(dir), fs::directory_iterator()) == std::ptrdiff_t(kNames.size());
}

// The error a call reports, empty when it succeeds.
template <class Call>
static std::error_code failure(Call&& call)
{
    try {
        call();
    } catch (const fs::filesystem_error& e) {
        return e.code();
    }
    return {};
}

template <class Call>
static bool refuses(Call&& call)
{
    try {
        call();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

int main(int argc, char** argv)
{
    const auto root = (argc > 1 ? fs::path(argv[1]) : fs::temp_directory_path()) /
                      ("vc_reader_publication_" + std::to_string(std::random_device()()));
    fs::create_directories(root);

    CHECK(renameFlagRefused(std::make_error_code(std::errc::invalid_argument)));
    CHECK(renameFlagRefused(std::error_code(ENOTSUP, std::generic_category())));
    CHECK(!renameFlagRefused(std::make_error_code(std::errc::file_exists)));
    CHECK(!renameFlagRefused(std::make_error_code(std::errc::io_error)));

    // Does this filesystem refuse the no-replace rename?
    put(root / "probe.tmp", "");
    const auto refused = failure([&] { publishPath(root / "probe.tmp", root / "probe", false); });
    CHECK(!refused || renameFlagRefused(refused));
    std::error_code ignored;
    fs::remove(root / "probe.tmp", ignored);
    fs::remove(root / "probe", ignored);
    std::printf("no-replace rename %s\n", refused ? "refused, so overwrite saves publish file by file" : "supported");

    // Without overwrite nothing existing is replaced, an empty directory included, and a
    // new destination is published atomically or not at all.
    fs::create_directory(root / "empty");
    stage(root / "a.tmp", "a");
    CHECK(failure([&] { publishPath(root / "a.tmp", root / "empty", false); }) == std::errc::file_exists);
    CHECK(failure([&] { publishDirectory(root / "a.tmp", root / "empty", false); }) == std::errc::file_exists);
    CHECK(fs::is_empty(root / "empty") && holds(root / "a.tmp", "a"));
    CHECK(failure([&] { publishDirectory(root / "a.tmp", root / "a", false); }) == refused);
    CHECK(refused ? !fs::exists(root / "a") && holds(root / "a.tmp", "a") : holds(root / "a", "a") && !fs::exists(root / "a.tmp"));

    // With overwrite a new destination is published, file by file where the rename is refused.
    stage(root / "b.tmp", "b");
    publishDirectory(root / "b.tmp", root / "b", true);
    CHECK(holds(root / "b", "b") && !fs::exists(root / "b.tmp"));

    // An interrupted save leaves an empty directory, or one without meta.json. A save with
    // overwrite completes either through the journal, on every filesystem.
    fs::create_directory(root / "c");
    stage(root / "c.tmp", "c");
    publishDirectory(root / "c.tmp", root / "c", true);
    CHECK(holds(root / "c", "c") && !fs::exists(root / "c.tmp"));
    fs::create_directory(root / "d");
    put(root / "d" / "x.tif", "old");
    stage(root / "d.tmp", "d");
    publishDirectory(root / "d.tmp", root / "d", true);
    CHECK(holds(root / "d", "d") && !fs::exists(root / "d.tmp"));

    // The file by file publisher itself, so these checks run on every filesystem.
    // Every file moves, meta.json last, and the staging directory goes.
    std::vector<std::string> order;
    const auto record = [&](const fs::path& from, const fs::path& to) {
        order.push_back(from.filename().string());
        fs::rename(from, to);
    };
    stage(root / "e.tmp", "e");
    CHECK(publishNewByFiles(root / "e.tmp", root / "e", record));
    CHECK(holds(root / "e", "e") && !fs::exists(root / "e.tmp"));
    CHECK(order.size() == kNames.size() && order.back() == "meta.json");

    // An existing destination, even an empty directory, is left alone.
    stage(root / "f.tmp", "f");
    CHECK(!publishNewByFiles(root / "f.tmp", root / "empty"));
    CHECK(fs::is_empty(root / "empty") && holds(root / "f.tmp", "f"));

    // A failed move undoes the publication whichever file fails: the files go back to
    // staging and the reservation is removed.
    for (std::size_t k = 0; k < kNames.size(); ++k) {
        std::size_t moves = 0;
        const auto fail = [&](const fs::path& from, const fs::path& to) {
            if (moves++ == k) throw fs::filesystem_error("injected", from, to, std::make_error_code(std::errc::io_error));
            fs::rename(from, to);
        };
        CHECK(failure([&] { publishNewByFiles(root / "f.tmp", root / "g", fail); }) == std::errc::io_error);
        CHECK(!fs::exists(root / "g") && holds(root / "f.tmp", "f"));
    }

    // When the files cannot move back either, the original error is reported, meta.json
    // is not left behind, and the retry with overwrite completes the destination.
    std::size_t moves = 0;
    const auto stuck = [&](const fs::path& from, const fs::path& to) {
        const auto n = moves++;
        if (n == kNames.size() - 1) throw fs::filesystem_error("injected", from, to, std::make_error_code(std::errc::io_error));
        if (n >= kNames.size()) throw fs::filesystem_error("injected", from, to, std::make_error_code(std::errc::no_space_on_device));
        fs::rename(from, to);
    };
    CHECK(failure([&] { publishNewByFiles(root / "f.tmp", root / "h", stuck); }) == std::errc::io_error);
    CHECK(fs::is_directory(root / "h") && !fs::exists(root / "h" / "meta.json") && fs::exists(root / "h" / "x.tif"));
    fs::remove_all(root / "f.tmp");
    stage(root / "h.tmp", "h");
    publishDirectory(root / "h.tmp", root / "h", true);
    CHECK(holds(root / "h", "h") && !fs::exists(root / "h.tmp"));

    // Only a directory of regular files outside the destination is published this way.
    stage(root / "i.tmp", "i");
    fs::create_directory(root / "i.tmp" / "sub");
    CHECK(refuses([&] { publishNewByFiles(root / "i.tmp", root / "i"); }) && !fs::exists(root / "i"));
    fs::remove(root / "i.tmp" / "sub");
    CHECK(refuses([&] { publishNewByFiles(root / "i.tmp", root / "i.tmp" / "i"); }) && holds(root / "i.tmp", "i"));

    fs::remove_all(root);
    std::puts("publication checks passed");
    return 0;
}
