// vcr::files publication: publishPath and publishDirectory stay strict, and a new
// directory is published whole even where the no-replace rename is refused (NFS,
// exFAT). Pass a directory on such a filesystem to run every check there.
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
using vcr::files::detail::publishNewDirectory;
using vcr::files::detail::publishStaged;
using vcr::files::detail::renameNewDirectory;
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
    std::printf("no-replace rename %s\n", refused ? "refused, so new directories are renamed whole" : "supported");

    // publishPath and publishDirectory never replace an existing destination, an empty
    // directory included, and publish a new one atomically or not at all.
    fs::create_directory(root / "empty");
    stage(root / "a.tmp", "a");
    CHECK(failure([&] { publishPath(root / "a.tmp", root / "empty", false); }) == std::errc::file_exists);
    CHECK(failure([&] { publishDirectory(root / "a.tmp", root / "empty", false); }) == std::errc::file_exists);
    CHECK(fs::is_empty(root / "empty") && holds(root / "a.tmp", "a"));
    CHECK(failure([&] { publishDirectory(root / "a.tmp", root / "a", false); }) == refused);
    CHECK(refused ? !fs::exists(root / "a") && holds(root / "a.tmp", "a") : holds(root / "a", "a") && !fs::exists(root / "a.tmp"));

    // publishNewDirectory publishes a new directory whole on every filesystem, and fails
    // on anything already there without changing either side.
    stage(root / "b.tmp", "b");
    publishNewDirectory(root / "b.tmp", root / "b");
    CHECK(holds(root / "b", "b") && !fs::exists(root / "b.tmp"));
    stage(root / "c.tmp", "c");
    CHECK(failure([&] { publishNewDirectory(root / "c.tmp", root / "b"); }) == std::errc::file_exists);
    CHECK(failure([&] { publishNewDirectory(root / "c.tmp", root / "empty"); }) == std::errc::file_exists);
    CHECK(holds(root / "b", "b") && fs::is_empty(root / "empty") && holds(root / "c.tmp", "c"));

    // A save with overwrite replaces what existed when it began, through the journal, and
    // fails on a segment another save published where nothing was when it began.
    CHECK(failure([&] { publishStaged(root / "c.tmp", root / "b", true, false); }) == std::errc::file_exists);
    CHECK(holds(root / "b", "b") && holds(root / "c.tmp", "c"));
    stage(root / "j.tmp", "j");
    publishStaged(root / "j.tmp", root / "b", true, true);
    CHECK(holds(root / "b", "j") && !fs::exists(root / "j.tmp"));
    stage(root / "k.tmp", "k");
    publishStaged(root / "k.tmp", root / "k", true, false);
    CHECK(holds(root / "k", "k") && !fs::exists(root / "k.tmp"));

    // The plain rename itself, so these run on every filesystem. It refuses an existing
    // destination, an empty directory included, before renaming.
    CHECK(failure([&] { renameNewDirectory(root / "c.tmp", root / "empty"); }) == std::errc::file_exists);
    CHECK(fs::is_empty(root / "empty") && holds(root / "c.tmp", "c"));

    // A competitor that publishes at the destination just before the rename wins, and
    // this save fails.
    const auto competitor = [&](const fs::path& from, const fs::path& to) {
        stage(to, "winner");
        fs::rename(from, to);
    };
    CHECK(failure([&] { renameNewDirectory(root / "c.tmp", root / "d", competitor); }) == std::errc::file_exists);
    CHECK(holds(root / "d", "winner") && holds(root / "c.tmp", "c"));

    // A rename reported as failed after it took effect (NFS) counts as published.
    const auto lost_reply = [&](const fs::path& from, const fs::path& to) {
        fs::rename(from, to);
        throw fs::filesystem_error("injected", from, to, std::make_error_code(std::errc::io_error));
    };
    renameNewDirectory(root / "c.tmp", root / "e", lost_reply);
    CHECK(holds(root / "e", "c") && !fs::exists(root / "c.tmp"));

    // A rename that fails without effect reports its error and leaves nothing behind.
    stage(root / "f.tmp", "f");
    const auto broken = [&](const fs::path& from, const fs::path& to) {
        throw fs::filesystem_error("injected", from, to, std::make_error_code(std::errc::io_error));
    };
    CHECK(failure([&] { renameNewDirectory(root / "f.tmp", root / "g", broken); }) == std::errc::io_error);
    CHECK(!fs::exists(root / "g") && holds(root / "f.tmp", "f"));

    // Only a staged directory outside the destination is published this way.
    put(root / "file.tmp", "x");
    CHECK(refuses([&] { renameNewDirectory(root / "file.tmp", root / "h"); }) && !fs::exists(root / "h"));
    CHECK(refuses([&] { renameNewDirectory(root / "f.tmp", root / "f.tmp" / "h"); }) && holds(root / "f.tmp", "f"));

    fs::remove_all(root);
    std::puts("publication checks passed");
    return 0;
}
