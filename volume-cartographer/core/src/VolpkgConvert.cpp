#include "vc/core/util/VolpkgConvert.hpp"

#include "utils/Json.hpp"
#include "vc/core/util/RemoteUrl.hpp"

#include <fstream>
#include <optional>
#include <sstream>
#include <vector>

namespace fs = std::filesystem;

namespace {

bool isRemoteScheme(const std::string& s)
{
    return s.rfind("s3://", 0) == 0
        || s.rfind("s3+", 0) == 0
        || s.rfind("http://", 0) == 0
        || s.rfind("https://", 0) == 0;
}

std::string ensureTrailingSlash(std::string s)
{
    if (s.empty() || s.back() != '/') s.push_back('/');
    return s;
}

std::string lastPrefixComponent(const std::string& prefix)
{
    auto s = prefix;
    if (!s.empty() && s.back() == '/') s.pop_back();
    auto pos = s.find_last_of('/');
    return pos == std::string::npos ? s : s.substr(pos + 1);
}

utils::Json parseConfig(const std::string& body)
{
    if (body.empty()) return utils::Json::object();
    return utils::Json::parse(body);
}

void appendBareEntry(utils::Json& arr, const std::string& location)
{
    arr.push_back(utils::Json(location));
}

void appendTaggedEntry(utils::Json& arr, const std::string& location,
                       const std::vector<std::string>& tags)
{
    if (tags.empty()) {
        arr.push_back(utils::Json(location));
        return;
    }
    auto obj = utils::Json::object();
    obj["location"] = location;
    auto t = utils::Json::array();
    for (const auto& s : tags) t.push_back(utils::Json(s));
    obj["tags"] = t;
    arr.push_back(obj);
}

std::string relPath(const fs::path& target, const fs::path& base)
{
    std::error_code ec;
    auto rel = fs::relative(target, base, ec);
    if (ec || rel.empty()) return target.string();
    return rel.string();
}

vc::VolpkgConvertResult writeOut(const fs::path& outFile, const utils::Json& out,
                                 std::string warning = {})
{
    std::error_code ec;
    if (!outFile.parent_path().empty()) {
        fs::create_directories(outFile.parent_path(), ec);
        if (ec) {
            return {false, "cannot create output directory " +
                            outFile.parent_path().string() + ": " + ec.message(), {}};
        }
    }
    std::ofstream of(outFile);
    if (!of) {
        return {false, "cannot open output file for writing: " + outFile.string(), {}};
    }
    of << out.dump(2) << '\n';
    if (!of) {
        return {false, "write failed for: " + outFile.string(), {}};
    }
    return {true, std::move(warning), outFile};
}

vc::VolpkgConvertResult convertLocal(const fs::path& root, const fs::path& outFile)
{
    if (!fs::is_directory(root)) {
        return {false, "input is not a directory: " + root.string(), {}};
    }

    const auto absRoot = fs::absolute(root);
    const auto absOutDir = fs::absolute(
        outFile.parent_path().empty() ? fs::current_path() : outFile.parent_path());

    utils::Json out = utils::Json::object();
    out["name"] = absRoot.filename().string();
    out["version"] = 1;

    std::string warning;
    const auto cfg = absRoot / "config.json";
    if (fs::exists(cfg)) {
        try {
            auto j = utils::Json::parse_file(cfg);
            if (j.contains("name") && j.at("name").is_string()) {
                const auto n = j.at("name").get_string();
                if (n != "NULL") out["name"] = n;
            }
        } catch (const std::exception& e) {
            warning = std::string("cannot parse ") + cfg.string() + ": " + e.what();
        }
    }

    auto volumes = utils::Json::array();
    auto segments = utils::Json::array();
    auto normalGrids = utils::Json::array();

    if (fs::is_directory(absRoot / "volumes")) {
        appendBareEntry(volumes, relPath(absRoot / "volumes", absOutDir));
    }

    std::optional<std::string> firstSegmentsLoc;
    for (const auto& d : {"paths", "traces", "export"}) {
        if (fs::is_directory(absRoot / d)) {
            const auto loc = relPath(absRoot / d, absOutDir);
            appendBareEntry(segments, loc);
            if (!firstSegmentsLoc) firstSegmentsLoc = loc;
        }
    }

    if (fs::is_directory(absRoot / "normal_grids")) {
        appendBareEntry(normalGrids, relPath(absRoot / "normal_grids", absOutDir));
    }

    if (fs::is_directory(absRoot / "normal3d")) {
        for (const auto& e : fs::directory_iterator(absRoot / "normal3d")) {
            if (e.is_directory()) {
                appendTaggedEntry(volumes, relPath(e.path(), absOutDir), {"normal3d"});
            }
        }
    }

    out["volumes"] = volumes;
    out["segments"] = segments;
    out["normal_grids"] = normalGrids;
    if (firstSegmentsLoc) out["output_segments"] = *firstSegmentsLoc;

    return writeOut(outFile, out, warning);
}

vc::VolpkgConvertResult convertRemote(const std::string& input, const fs::path& outFile)
{
    (void)input;
    (void)outFile;
    return {false, "remote volpkg conversion is not supported in this branch", {}};
}

}

namespace vc {

VolpkgConvertResult convertVolpkg(const std::string& input, const fs::path& outFile)
{
    try {
        if (isRemoteScheme(input)) return convertRemote(input, outFile);
        return convertLocal(input, outFile);
    } catch (const std::exception& e) {
        return {false, std::string("unhandled exception: ") + e.what(), {}};
    } catch (...) {
        return {false, "unknown unhandled exception", {}};
    }
}

}
