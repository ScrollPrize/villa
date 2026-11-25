// How vc_objrefine puts its offset report in place
// (apps/src/vc_objrefine_report_publish.hpp): files beside or inside the
// report folder that are not the report are never removed, overwrite is
// honored when the report is published, and a failed replacement leaves the
// earlier report whole. The tool's end-to-end runs are in
// test_objrefine_offset_report_cli.cpp.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "vc_test.hpp"

#include "../../apps/src/vc_objrefine_report_publish.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace fs = std::filesystem;
using vc_objrefine::publishReport;
using vc_objrefine::reportFolderRefusal;

namespace {

const std::vector<std::string> REPORT_FILES{"offset.tif", "opacity.tif", "report.json"};

std::vector<std::string> listing(const fs::path& dir)
{
    std::vector<std::string> names;
    for (const auto& e : fs::directory_iterator(dir))
        names.push_back(e.path().filename().string());
    std::sort(names.begin(), names.end());
    return names;
}

void put(const fs::path& p, const std::string& text)
{
    fs::create_directories(p.parent_path());
    std::ofstream f(p, std::ios::binary);
    f << text;
    REQUIRE(f.good());
}

std::string slurp(const fs::path& p)
{
    std::ifstream f(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), {});
}

// Writes a report whose three files all say `tag`.
std::function<void(const fs::path&)> writer(const std::string& tag)
{
    return [tag](const fs::path& dir) {
        for (const auto& name : REPORT_FILES)
            put(dir / name, tag);
    };
}

bool holdsReport(const fs::path& dir, const std::string& tag)
{
    if (!fs::is_directory(dir) || listing(dir) != REPORT_FILES)
        return false;
    for (const auto& name : REPORT_FILES)
        if (slurp(dir / name) != tag)
            return false;
    return true;
}

struct TempDir {
    fs::path path;
    TempDir()
    {
        static int counter = 0;
        path = fs::temp_directory_path() /
               ("vc_objrefine_publish_" + std::to_string(::getpid()) + "_" + std::to_string(counter++));
        fs::remove_all(path);
        fs::create_directories(path);
    }
    ~TempDir() { std::error_code ec; fs::remove_all(path, ec); }
};

}  // namespace

TEST_CASE("a new report lands whole and leaves its neighbours alone")
{
    TempDir tmp;
    const fs::path dir = tmp.path / "measured";
    // A sibling at the name the tool once used for staging, and another file.
    put(tmp.path / ".measured.tmp" / "keep.txt", "mine");
    put(tmp.path / "notes.txt", "notes");

    CHECK(publishReport(dir, false, writer("one")).empty());
    CHECK(holdsReport(dir, "one"));
    CHECK(slurp(tmp.path / ".measured.tmp" / "keep.txt") == "mine");
    CHECK(listing(tmp.path) == std::vector<std::string>{".measured.tmp", "measured", "notes.txt"});
}

TEST_CASE("an earlier report is replaced only with overwrite")
{
    TempDir tmp;
    const fs::path dir = tmp.path / "measured";
    REQUIRE(publishReport(dir, true, writer("one")).empty());

    CHECK_THROWS_AS(publishReport(dir, false, writer("two")), std::runtime_error);
    CHECK(holdsReport(dir, "one"));
    CHECK(listing(tmp.path) == std::vector<std::string>{"measured"});

    CHECK(publishReport(dir, true, writer("two")).empty());
    CHECK(holdsReport(dir, "two"));
    CHECK(listing(tmp.path) == std::vector<std::string>{"measured"});
}

TEST_CASE("files that arrive while the report is measured are kept, and the report refused")
{
    TempDir tmp;
    const fs::path dir = tmp.path / "measured";
    fs::create_directories(dir);
    // The folder was empty when the run started; a file appears before the
    // report is published (here, while it is being written).
    const auto write = [&](const fs::path& staged) {
        put(dir / "added.txt", "someone else's");
        writer("one")(staged);
    };
    CHECK_THROWS_AS(publishReport(dir, true, write), std::runtime_error);
    CHECK(listing(dir) == std::vector<std::string>{"added.txt"});
    CHECK(slurp(dir / "added.txt") == "someone else's");
    CHECK(listing(tmp.path) == std::vector<std::string>{"measured"});

    // A report that appears meanwhile is not overwritten without overwrite.
    fs::remove(dir / "added.txt");
    const auto writeRace = [&](const fs::path& staged) {
        writer("other run")(dir);
        writer("one")(staged);
    };
    CHECK_THROWS_AS(publishReport(dir, false, writeRace), std::runtime_error);
    CHECK(holdsReport(dir, "other run"));
    CHECK(listing(tmp.path) == std::vector<std::string>{"measured"});
}

TEST_CASE("only regular report files count as a report")
{
    TempDir tmp;
    const fs::path dir = tmp.path / "measured";
    fs::create_directories(dir / "report.json");
    put(dir / "report.json" / "inside.txt", "a folder named like a report file");
    CHECK_FALSE(reportFolderRefusal(dir, true, dir).empty());
    CHECK_THROWS_AS(publishReport(dir, true, writer("one")), std::runtime_error);
    CHECK(slurp(dir / "report.json" / "inside.txt") == "a folder named like a report file");

#ifndef _WIN32
    const fs::path linked = tmp.path / "linked";
    fs::create_directories(linked);
    put(tmp.path / "elsewhere.json", "outside");
    fs::create_symlink(tmp.path / "elsewhere.json", linked / "report.json");
    CHECK_FALSE(reportFolderRefusal(linked, true, linked).empty());
    const fs::path linkFolder = tmp.path / "link-to-folder";
    fs::create_directory_symlink(linked, linkFolder);
    CHECK_FALSE(reportFolderRefusal(linkFolder, true, linkFolder).empty());
#endif
}

TEST_CASE("a failed replacement keeps the earlier report whole")
{
    TempDir tmp;
    const fs::path dir = tmp.path / "measured";
    REQUIRE(publishReport(dir, true, writer("one")).empty());

    // The new report cannot be moved in: the old one is put back.
    const auto failInstall = [&](const fs::path& from, const fs::path& to) {
        if (to == dir && from.filename().string().find(".new-") != std::string::npos)
            throw std::runtime_error("simulated failure");
        fs::rename(from, to);
    };
    CHECK_THROWS_AS(publishReport(dir, true, writer("two"), failInstall), std::runtime_error);
    CHECK(holdsReport(dir, "one"));
    CHECK(listing(tmp.path) == std::vector<std::string>{"measured"});

    // The old one cannot be moved aside: nothing changes.
    const auto failAside = [&](const fs::path& from, const fs::path& to) {
        if (from == dir)
            throw std::runtime_error("simulated failure");
        fs::rename(from, to);
    };
    CHECK_THROWS_AS(publishReport(dir, true, writer("two"), failAside), std::runtime_error);
    CHECK(holdsReport(dir, "one"));
    CHECK(listing(tmp.path) == std::vector<std::string>{"measured"});

    // Writing the new report fails: nothing changes.
    const auto failWrite = [](const fs::path& staged) {
        put(staged / "offset.tif", "partial");
        throw std::runtime_error("disk full");
    };
    CHECK_THROWS_AS(publishReport(dir, true, failWrite), std::runtime_error);
    CHECK(holdsReport(dir, "one"));
    CHECK(listing(tmp.path) == std::vector<std::string>{"measured"});
}

#ifndef _WIN32
TEST_CASE("an old report that cannot be deleted is replaced, never mixed")
{
    if (::geteuid() == 0) {
        MESSAGE("skipped: permissions do not bind root");
        return;
    }
    TempDir tmp;
    const fs::path dir = tmp.path / "measured";
    REQUIRE(publishReport(dir, true, writer("one")).empty());
    // A read-only report folder: its files cannot be removed one by one, but
    // the folder can still be moved within its parent.
    fs::permissions(dir, fs::perms::owner_write | fs::perms::group_write | fs::perms::others_write,
                    fs::perm_options::remove);
    const std::string warning = publishReport(dir, true, writer("two"));
    CHECK_FALSE(warning.empty());
    CHECK(holdsReport(dir, "two"));
    // The old copy is left whole under a hidden name, and named in the warning.
    int oldCopies = 0;
    for (const auto& e : fs::directory_iterator(tmp.path)) {
        if (e.path().filename().string().rfind(".measured.old-", 0) == 0) {
            ++oldCopies;
            CHECK(holdsReport(e.path(), "one"));
            CHECK(warning.find(e.path().string()) != std::string::npos);
            fs::permissions(e.path(), fs::perms::owner_all, fs::perm_options::add);
        }
    }
    CHECK(oldCopies == 1);
}
#endif
