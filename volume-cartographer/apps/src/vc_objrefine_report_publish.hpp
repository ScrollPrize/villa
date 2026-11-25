// How vc_objrefine puts its offset report folder in place. Header-only so the
// tool and core/test/test_objrefine_report_publish.cpp run the same code; see
// ObjAlphaCompRefinement.cpp for the callers.
//
// The rules: a report folder holds only offset.tif, opacity.tif and
// report.json as regular files; anything else in it, or beside it, is never
// deleted; an earlier report is replaced only with overwrite, checked when the
// new report is published (not only when the run starts); and the earlier
// report stays whole at its path until the new one is complete, so a failed
// replacement leaves the old report, never a mix.
#pragma once

#include <cstdio>
#include <filesystem>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>

namespace vc_objrefine {

namespace report_fs = std::filesystem;

inline bool isReportFileName(const report_fs::path& name)
{
    return name == "offset.tif" || name == "opacity.tif" || name == "report.json";
}

// Why `dir` may not receive a report, or empty when it may: it must be
// missing, or a real folder (not a link) holding only report files as
// regular files; holding any of them, it needs overwrite. `shown` names the
// folder in the message.
inline std::string reportFolderRefusal(const report_fs::path& dir, bool overwrite,
                                       const report_fs::path& shown)
{
    std::error_code ec;
    const auto status = report_fs::symlink_status(dir, ec);
    if (ec && ec != std::errc::no_such_file_or_directory) {
        return "cannot read " + shown.string() + ": " + ec.message();
    }
    if (!report_fs::exists(status)) {
        return {};
    }
    if (report_fs::is_symlink(status)) {
        return shown.string() + " is a link; write the report to a new or empty folder";
    }
    if (!report_fs::is_directory(status)) {
        return shown.string() + " exists and is not a folder";
    }
    bool hasReport = false;
    report_fs::directory_iterator it(dir, ec);
    for (; !ec && it != report_fs::directory_iterator(); it.increment(ec)) {
        const report_fs::path name = it->path().filename();
        std::error_code typeEc;
        const bool regular = it->symlink_status(typeEc).type() == report_fs::file_type::regular;
        if (typeEc || !regular || !isReportFileName(name)) {
            return shown.string() + " holds " + name.string() +
                   "; write the report to a new or empty folder, not into a segment";
        }
        hasReport = true;
    }
    if (ec) {
        return "cannot read " + shown.string() + ": " + ec.message();
    }
    if (hasReport && !overwrite) {
        return shown.string() + " already holds a report and overwrite is false";
    }
    return {};
}

inline bool pathExists(const report_fs::path& p)
{
    std::error_code ec;
    return report_fs::exists(report_fs::symlink_status(p, ec));
}

// A free hidden name beside `path`: ".<name>.<tag>-<random>". Nothing is
// created, so the caller claims it (create_directory or rename).
inline report_fs::path freeSibling(const report_fs::path& path, const std::string& tag)
{
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    const report_fs::path parent = path.parent_path();
    const std::string base = "." + path.filename().string() + "." + tag + "-";
    for (int attempt = 0; attempt < 100; ++attempt) {
        char suffix[17];
        std::snprintf(suffix, sizeof suffix, "%016llx", static_cast<unsigned long long>(rng()));
        report_fs::path candidate = parent / (base + suffix);
        if (!pathExists(candidate)) {
            return candidate;
        }
    }
    throw std::runtime_error("cannot find a free name beside " + path.string());
}

using RenameFn = std::function<void(const report_fs::path&, const report_fs::path&)>;

inline void plainRename(const report_fs::path& from, const report_fs::path& to)
{
    report_fs::rename(from, to);
}

// Writes a report into `dir` through `write`, which fills the staging folder
// it is given. The staging folder has a fresh name beside `dir`, so no other
// file is ever removed or overwritten. The folder at `dir`, if any, is moved
// aside whole, checked (reportFolderRefusal), and either deleted once the new
// report is in place or moved back if anything fails. Throws on failure with
// the earlier state at `dir`; returns a warning (empty when none) when the
// new report is in place but the moved-aside old one could not be deleted.
// `rename` lets tests make a step fail.
inline std::string publishReport(const report_fs::path& dir, bool overwrite,
                                 const std::function<void(const report_fs::path&)>& write,
                                 const RenameFn& rename = plainRename)
{
    // Refuse early without touching anything; checked again on what is
    // actually there when the report is published.
    if (std::string why = reportFolderRefusal(dir, overwrite, dir); !why.empty()) {
        throw std::runtime_error(why);
    }

    report_fs::path staged;
    for (int attempt = 0; staged.empty(); ++attempt) {
        const report_fs::path candidate = freeSibling(dir, "new");
        if (report_fs::create_directory(candidate)) {
            staged = candidate;
        } else if (attempt >= 10) {
            throw std::runtime_error("cannot create a staging folder beside " + dir.string());
        }
    }
    const auto dropStaged = [&] {
        std::error_code ec;
        report_fs::remove_all(staged, ec);
    };
    try {
        write(staged);
    } catch (...) {
        dropStaged();
        throw;
    }

    report_fs::path aside;
    if (pathExists(dir)) {
        aside = freeSibling(dir, "old");
        try {
            rename(dir, aside);
        } catch (const std::exception& e) {
            dropStaged();
            throw std::runtime_error("cannot move the earlier report in " + dir.string() +
                                     " aside, so it was kept: " + e.what());
        }
    }
    const auto putBack = [&]() -> std::string {
        if (aside.empty()) {
            return {};
        }
        try {
            rename(aside, dir);
            return {};
        } catch (const std::exception& e) {
            return "; the earlier report could not be moved back and is in " + aside.string() +
                   " (" + e.what() + ")";
        }
    };
    if (!aside.empty()) {
        // Whatever arrived since the run started (another file, a report
        // with overwrite false) is refused here, and put back untouched.
        if (std::string why = reportFolderRefusal(aside, overwrite, dir); !why.empty()) {
            const std::string where = putBack();
            dropStaged();
            throw std::runtime_error(why + where);
        }
    }
    try {
        rename(staged, dir);
    } catch (const std::exception& e) {
        const std::string where = putBack();
        dropStaged();
        throw std::runtime_error("cannot move the new report into " + dir.string() + ": " + e.what() +
                                 (aside.empty() || !where.empty() ? where : "; the earlier report was kept"));
    }
    if (!aside.empty()) {
        std::error_code ec;
        report_fs::remove_all(aside, ec);
        if (ec) {
            return "the earlier report was replaced, but its old copy in " + aside.string() +
                   " could not be deleted: " + ec.message();
        }
    }
    return {};
}

}  // namespace vc_objrefine
