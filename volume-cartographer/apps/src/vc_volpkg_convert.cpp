#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "utils/Json.hpp"
#include "vc/core/cache/HttpMetadataFetcher.hpp"
#include "vc/core/util/RemoteUrl.hpp"

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
    try {
        return utils::Json::parse(body);
    } catch (...) {
        return utils::Json::object();
    }
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

int convertLocal(const fs::path& root, const fs::path& outFile)
{
    if (!fs::is_directory(root)) {
        std::cerr << "input is not a directory: " << root << "\n";
        return 2;
    }

    utils::Json out = utils::Json::object();
    out["name"] = root.filename().string();
    out["version"] = 1;

    const auto cfg = root / "config.json";
    if (fs::exists(cfg)) {
        auto j = utils::Json::parse_file(cfg);
        if (j.contains("name") && j.at("name").is_string()) {
            const auto n = j.at("name").get_string();
            if (n != "NULL") out["name"] = n;
        }
    }

    auto volumes = utils::Json::array();
    auto segments = utils::Json::array();
    auto normalGrids = utils::Json::array();

    if (fs::is_directory(root / "volumes")) {
        appendBareEntry(volumes, (root / "volumes").string());
    }

    std::optional<std::string> firstSegmentsLoc;
    for (const auto& d : {"paths", "traces", "export"}) {
        if (fs::is_directory(root / d)) {
            const auto loc = (root / d).string();
            appendBareEntry(segments, loc);
            if (!firstSegmentsLoc) firstSegmentsLoc = loc;
        }
    }

    if (fs::is_directory(root / "normal_grids")) {
        appendBareEntry(normalGrids, (root / "normal_grids").string());
    }

    if (fs::is_directory(root / "normal3d")) {
        for (const auto& e : fs::directory_iterator(root / "normal3d")) {
            if (e.is_directory()) {
                appendTaggedEntry(volumes, e.path().string(), {"normal3d"});
            }
        }
    }

    out["volumes"] = volumes;
    out["segments"] = segments;
    out["normal_grids"] = normalGrids;
    if (firstSegmentsLoc) out["output_segments"] = *firstSegmentsLoc;

    fs::create_directories(outFile.parent_path());
    std::ofstream of(outFile);
    if (!of) { std::cerr << "cannot open output: " << outFile << "\n"; return 3; }
    of << out.dump(2) << '\n';
    std::cout << "wrote " << outFile << "\n";
    return 0;
}

int convertRemote(const std::string& input, const fs::path& outFile)
{
    auto resolved = vc::resolveRemoteUrl(input);
    auto httpsBase = ensureTrailingSlash(resolved.httpsUrl);
    // Preserve the user-supplied form (s3:// or https://) for output entries.
    auto displayBase = ensureTrailingSlash(input);

    vc::cache::HttpAuth auth;
    if (resolved.useAwsSigv4) auth = vc::cache::loadAwsCredentials();

    vc::cache::S3ListResult listing;
    try {
        listing = vc::cache::s3ListObjects(httpsBase, auth);
    } catch (const std::exception& e) {
        std::cerr << "list failed for " << httpsBase << ": " << e.what() << "\n";
        return 4;
    }
    if (listing.authError) {
        std::cerr << "auth error: " << listing.errorMessage << "\n";
        return 4;
    }

    utils::Json out = utils::Json::object();
    out["name"] = lastPrefixComponent(displayBase);
    out["version"] = 1;

    auto cfgUrl = httpsBase + "config.json";
    try {
        auto body = vc::cache::httpGetString(cfgUrl, auth);
        auto j = parseConfig(body);
        if (j.contains("name") && j.at("name").is_string()) {
            const auto n = j.at("name").get_string();
            if (n != "NULL") out["name"] = n;
        }
    } catch (...) {}

    auto volumes = utils::Json::array();
    auto segments = utils::Json::array();
    auto normalGrids = utils::Json::array();

    // s3ListObjects returns relative subprefix names with trailing slashes stripped
    // (e.g. "volumes", "paths"), so we just look up by exact name.
    auto hasSubprefix = [&](const std::string& name) {
        for (const auto& p : listing.prefixes) {
            if (p == name) return true;
        }
        return false;
    };

    if (hasSubprefix("volumes")) {
        appendBareEntry(volumes, displayBase + "volumes/");
    }

    std::optional<std::string> firstSegmentsLoc;
    for (const auto& d : {"paths", "traces", "export"}) {
        if (hasSubprefix(d)) {
            const auto loc = displayBase + d + "/";
            appendBareEntry(segments, loc);
            if (!firstSegmentsLoc) firstSegmentsLoc = loc;
        }
    }

    if (hasSubprefix("normal_grids")) {
        appendBareEntry(normalGrids, displayBase + "normal_grids/");
    }

    if (hasSubprefix("normal3d")) {
        try {
            auto sub = vc::cache::s3ListObjects(httpsBase + "normal3d/", auth);
            for (const auto& subName : sub.prefixes) {
                appendTaggedEntry(volumes,
                    displayBase + "normal3d/" + subName + "/", {"normal3d"});
            }
        } catch (...) {}
    }

    out["volumes"] = volumes;
    out["segments"] = segments;
    out["normal_grids"] = normalGrids;
    if (firstSegmentsLoc) out["output_segments"] = *firstSegmentsLoc;

    fs::create_directories(outFile.parent_path());
    std::ofstream of(outFile);
    if (!of) { std::cerr << "cannot open output: " << outFile << "\n"; return 3; }
    of << out.dump(2) << '\n';
    std::cout << "wrote " << outFile << "\n";
    return 0;
}

}

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "usage: " << argv[0] << " <input> <output.volpkg.json>\n"
                  << "  <input>: local volpkg directory or s3://, https:// URL\n";
        return 2;
    }
    const std::string input = argv[1];
    const fs::path outFile = argv[2];

    if (isRemoteScheme(input)) return convertRemote(input, outFile);
    return convertLocal(input, outFile);
}
