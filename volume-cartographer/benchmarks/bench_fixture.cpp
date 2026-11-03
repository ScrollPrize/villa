#include "bench_fixture.hpp"

#include "vc/core/util/QuadSurface.hpp"

#ifdef VC_BENCH_VC_READER
#include <vc_reader/cache.hpp>
#include <vc_reader/zarr.hpp>
#endif

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <unistd.h>

#ifndef VC_BENCH_DEFAULT_CONFIG
#define VC_BENCH_DEFAULT_CONFIG "benchmarks.json"
#endif

namespace fs = std::filesystem;
using nlohmann::json;

namespace vcbench {

namespace {

// The cache's record of where each remote item's copy is: {"<url>": "<path>"}.
// The prefetch (vc_reader trees) writes it; every tree reads it, so trees without
// vc_reader find the cached copies without knowing vc_reader's cache layout.
constexpr auto kIndexFile = "benchmark-index.json";

json readJson(const fs::path& file)
{
    std::ifstream in(file);
    if (!in)
        throw std::runtime_error("cannot read " + file.string());
    try {
        return json::parse(in);
    } catch (const std::exception& e) {
        throw std::runtime_error(file.string() + ": " + e.what());
    }
}

std::string kebab(std::string key)
{
    std::replace(key.begin(), key.end(), '_', '-');
    return key;
}

std::string text(const json& value)
{
    if (value.is_string())
        return value.get<std::string>();
    if (value.is_boolean())
        return value.get<bool>() ? "1" : "0";
    if (value.is_array()) {
        std::string out;
        for (const auto& v : value)
            out += (out.empty() ? "" : ",") + text(v);
        return out;
    }
    return value.dump();
}

fs::path resolveAgainst(const fs::path& base, const std::string& path)
{
    const fs::path p(path);
    return p.is_absolute() ? p : (base / p).lexically_normal();
}

void writeVc3dCacheConfig(const fs::path& configDir, const fs::path& root)
{
    fs::create_directories(configDir);
    std::ofstream out(configDir / "cache.json");
    out << json{{"version", 1}, {"root", root.string()}}.dump() << "\n";
    if (!out)
        throw std::runtime_error("cannot write " + (configDir / "cache.json").string());
}

// Hard-links (copies across file systems) every file under `from` into `to`.
std::size_t linkTree(const fs::path& from, const fs::path& to)
{
    std::size_t files = 0;
    fs::create_directories(to);
    for (auto it = fs::recursive_directory_iterator(from); it != fs::recursive_directory_iterator(); ++it) {
        const auto target = to / fs::relative(it->path(), from);
        if (it->is_directory()) {
            fs::create_directories(target);
        } else if (it->is_regular_file()) {
            std::error_code ec;
            fs::create_hard_link(it->path(), target, ec);
            if (ec)
                fs::copy_file(it->path(), target);
            ++files;
        }
    }
    return files;
}

bool isLevelDir(const fs::directory_entry& entry)
{
    const auto name = entry.path().filename().string();
    return entry.is_directory() && !name.empty() && std::all_of(name.begin(), name.end(), ::isdigit);
}

#ifdef VC_BENCH_VC_READER
fs::path cachedCopy(const fs::path& root, const std::string& url)
{
    return vcr::zarr::decodedCachePath(root, url);
}

// Downloads every file under `url` into the cache (files already there are reused).
void downloadPrefix(const fs::path& root, const std::string& url)
{
    vcr::zarr::OpenOptions options;
    options.cacheDirs = {root.string()};
    options.cacheWrites = true;
    options.access = vcr::zarr::resolveAccess(url);
    int lastReported = -1;
    const auto result = vcr::cache::downloadPrefix(url, options, 16, [&](const vcr::cache::DownloadProgress& p) {
        if (p.status == "downloading" && (p.completedFiles == p.totalFiles || p.completedFiles - lastReported >= 64)) {
            lastReported = p.completedFiles;
            std::fprintf(stderr, "  %s: %d/%d files, %llu MB\n", url.c_str(), p.completedFiles, p.totalFiles,
                         static_cast<unsigned long long>(p.completedBytes >> 20));
        }
    });
    if (!result.error.empty())
        throw std::runtime_error("downloading " + url + ": " + result.error);
    if (result.progress.failedFiles > 0)
        throw std::runtime_error("downloading " + url + ": " + std::to_string(result.progress.failedFiles) +
                                 " files failed");
}
#endif

} // namespace

// ---------------------------------------------------------------------------
// Fixture

Fixture::Fixture(int argc, char** argv, std::string benchmark) : args_(argc, argv), benchmark_(std::move(benchmark))
{
    configPath_ = fs::absolute(args_.str("config", VC_BENCH_DEFAULT_CONFIG));
    manifest_ = readJson(configPath_);
    const auto base = configPath_.parent_path();
    cacheRoot_ = fs::absolute(resolveAgainst(base, manifest_.value("cache", std::string("cache"))));

    const auto benchmarks = manifest_.value("benchmarks", json::object());
    json cases = benchmarks.value(benchmark_, json::array());
    if (cases.is_object())
        cases = json::array({cases});
    if (args_.flag("list-cases")) {
        listOnly_ = true;
        for (const auto& c : cases)
            std::printf("%s\n", c.value("name", std::string()).c_str());
        return;
    }
    if (cases.empty())
        throw std::invalid_argument(configPath_.string() + " has no cases for benchmarks." + benchmark_);
    const auto wanted = args_.str("case");
    if (wanted.empty() && cases.size() > 1) {
        std::string names;
        for (const auto& c : cases)
            names += " " + c.value("name", std::string("?"));
        throw std::invalid_argument("--case is required, one of:" + names);
    }
    for (const auto& c : cases) {
        if (!c.is_object())
            throw std::invalid_argument("benchmarks." + benchmark_ + " cases must be objects");
        if (wanted.empty() || c.value("name", std::string()) == wanted) {
            case_ = c;
            break;
        }
    }
    if (case_.is_null())
        throw std::invalid_argument("benchmarks." + benchmark_ + " has no case named '" + wanted + "'");
    caseName_ = case_.value("name", std::string("default"));
    usedCaseKeys_ = {"name", "description"};

    prefetch_ = args_.flag("prefetch");
    cacheState_ = prefetch_ ? "prefetch" : str("cache_state", "cached");
    if (cacheState_ != "cached" && cacheState_ != "cold" && cacheState_ != "partial" && cacheState_ != "prefetch")
        throw std::invalid_argument("cache_state must be cached, cold or partial");
    keepScratch_ = args_.flag("keep-scratch");
    backend_ = str("backend", "cpu");
    if (backend_ != "cpu" && backend_ != "cuda")
        throw std::invalid_argument("--backend must be cpu or cuda");
#ifdef VC_BENCH_VC_READER
    pipeline_ = "vc_reader-" + backend_;
#else
    pipeline_ = "chunkcache";
    if (backend_ != "cpu")
        throw std::invalid_argument("this tree's pipeline (ChunkCache) has no --backend but cpu");
    if (cacheState_ != "cached")
        throw std::invalid_argument("--prefetch and cache states cold and partial need a vc_reader build; this "
                                    "tree only reads the cached copies");
#endif
    jsonPath_ = args_.str("json");
    dumpDir_ = args_.str("dump");

    config_["config"] = configPath_.string();
    config_["case"] = caseName_;
    config_["cache"] = cacheRoot_.string();
    config_["cache_state"] = cacheState_;
}

Fixture::~Fixture()
{
    if (!scratch_.empty() && !keepScratch_) {
        std::error_code ec;
        fs::remove_all(scratch_, ec);
    }
}

const json* Fixture::lookup(const std::string& key)
{
    if (case_.contains(key)) {
        usedCaseKeys_.insert(key);
        return &case_[key];
    }
    if (manifest_.contains("defaults") && manifest_["defaults"].contains(key))
        return &manifest_["defaults"][key];
    return nullptr;
}

std::string Fixture::str(const std::string& key, const std::string& fallback)
{
    const auto* value = lookup(key);  // marks a case key used even when overridden
    std::string out = fallback;
    if (const auto cli = args_.get(kebab(key)))
        out = *cli;
    else if (value)
        out = text(*value);
    // Every parameter's effective value goes into the report, so results with
    // different workloads can be told apart (compare.py lists the differences).
    if (!out.empty())
        config_[key] = out;
    return out;
}

double Fixture::num(const std::string& key, double fallback)
{
    const auto value = str(key);
    double out = fallback;
    if (!value.empty()) {
        try {
            out = std::stod(value);
        } catch (const std::exception&) {
            throw std::invalid_argument(key + ": not a number: " + value);
        }
    }
    std::ostringstream text;
    text << out;
    config_[key] = text.str();
    return out;
}

bool Fixture::flag(const std::string& key, bool fallback)
{
    const auto value = str(key);
    const bool out = value.empty() ? fallback : value != "0" && value != "false";
    config_[key] = out ? "1" : "0";
    return out;
}

std::vector<double> Fixture::numbers(const std::string& key)
{
    std::vector<double> out;
    const auto value = str(key);
    std::size_t at = 0;
    while (at < value.size()) {
        const auto comma = value.find(',', at);
        const auto part = value.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
        try {
            out.push_back(std::stod(part));
        } catch (const std::exception&) {
            throw std::invalid_argument(key + ": not a list of numbers: " + value);
        }
        if (comma == std::string::npos)
            break;
        at = comma + 1;
    }
    return out;
}

void Fixture::rejectUnused() const
{
    args_.rejectUnused();
    for (const auto& [key, value] : case_.items())
        if (!usedCaseKeys_.count(key))
            throw std::invalid_argument("benchmarks." + benchmark_ + " case '" + caseName_ + "': unknown key '" +
                                        key + "'");
}

const DataItem& Fixture::item(const std::string& kind, const std::string& name)
{
    const auto key = kind + "/" + name;
    // The report names each item by its source (after str() noted the manifest name).
    const auto note = [this, &kind](const DataItem& item) -> const DataItem& {
        config_[kind.substr(0, kind.size() - 1)] = item.url.empty() ? item.path.string() : item.url;
        return item;
    };
    if (const auto it = items_.find(key); it != items_.end())
        return note(it->second);
    const auto section = manifest_.value(kind, json::object());
    if (!section.contains(name))
        throw std::invalid_argument(configPath_.string() + ": no " + kind + " entry '" + name + "'");
    const auto& entry = section[name];
    DataItem item;
    item.kind = kind;
    item.name = name;
    item.url = entry.value("url", std::string());
    item.volume = entry.value("volume", std::string());
    item.scale = entry.value("scale", 1.0);
    if (kind != "volumes" && item.volume.empty())
        throw std::invalid_argument(kind + "." + name + ": \"volume\" is required");

    if (entry.contains("path")) {
        // A local copy, read in place (nothing to prefetch).
        item.path = resolveAgainst(configPath_.parent_path(), entry["path"].get<std::string>());
    } else if (item.url.empty()) {
        throw std::invalid_argument(kind + "." + name + ": \"url\" or \"path\" is required");
    } else if (prefetch_) {
#ifdef VC_BENCH_VC_READER
        item.path = cachedCopy(cacheRoot_, item.url);
        // A volume's chunks are fetched by replaying the scenarios; files are downloaded.
        if (kind != "volumes") {
            std::fprintf(stderr, "prefetch: %s.%s\n", kind.c_str(), name.c_str());
            downloadPrefix(cacheRoot_, item.url);
        }
        recordInIndex(item);
#endif
    } else {
        const auto indexFile = cacheRoot_ / kIndexFile;
        const auto index = fs::exists(indexFile) ? readJson(indexFile) : json::object();
        if (index.contains(item.url))
            item.path = index[item.url].get<std::string>();
        // A cold run streams the volume and needs no cached copy of it.
        else if (!(kind == "volumes" && cacheState_ == "cold"))
            throw std::runtime_error(kind + "." + name + " (" + item.url + ") is not in the cache " +
                                     cacheRoot_.string() + ": run benchmarks/prefetch.sh");
    }
    const bool streamed = kind == "volumes" && (prefetch_ || cacheState_ == "cold");
    if (!streamed && !fs::exists(item.path))
        throw std::runtime_error(kind + "." + name + ": " + item.path.string() + " does not exist");
    return note(items_.emplace(key, std::move(item)).first->second);
}

void Fixture::recordInIndex(const DataItem& item)
{
    fs::create_directories(cacheRoot_);
    const auto indexFile = cacheRoot_ / kIndexFile;
    auto index = fs::exists(indexFile) ? readJson(indexFile) : json::object();
    if (index.value(item.url, std::string()) == item.path.string())
        return;
    index[item.url] = item.path.string();
    const auto tmp = indexFile.string() + ".tmp";
    {
        std::ofstream out(tmp);
        out << index.dump(2) << "\n";
        if (!out)
            throw std::runtime_error("cannot write " + tmp);
    }
    fs::rename(tmp, indexFile);
}

const DataItem& Fixture::segment()
{
    const auto name = str("segment");
    if (name.empty())
        throw std::invalid_argument("benchmarks." + benchmark_ + " case '" + caseName_ + "' needs a \"segment\"");
    return item("segments", name);
}

const DataItem& Fixture::fibers()
{
    const auto name = str("fibers");
    if (name.empty())
        throw std::invalid_argument("benchmarks." + benchmark_ + " case '" + caseName_ + "' needs \"fibers\"");
    return item("fibers", name);
}

const DataItem& Fixture::volume()
{
    auto name = str("volume");
    if (name.empty() && !str("segment").empty())
        name = segment().volume;
    if (name.empty() && !str("fibers").empty())
        name = fibers().volume;
    if (name.empty())
        throw std::invalid_argument("benchmarks." + benchmark_ + " case '" + caseName_ +
                                    "' names no volume, segment or fibers");
    return item("volumes", name);
}

std::shared_ptr<QuadSurface> Fixture::loadSegment()
{
    if (!segment_) {
        const auto& s = segment();
        segment_ = loadSurface(s.path, float(s.scale));
    }
    return segment_;
}

std::vector<Fiber> Fixture::loadFibers(int count)
{
    const auto& source = fibers();
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(source.path))
        if (entry.path().extension() == ".json")
            files.push_back(entry.path());
    std::sort(files.begin(), files.end());
    std::vector<Fiber> out;
    for (const auto& file : files) {
        std::ifstream in(file);
        const auto doc = json::parse(in, nullptr, false);
        if (doc.is_discarded() || !doc.contains("line_points") || !doc["line_points"].is_array())
            continue;
        Fiber fiber{file.filename().string(), {}};
        for (const auto& p : doc["line_points"])
            if (p.is_array() && p.size() == 3)
                fiber.points.emplace_back(p[0].get<double>() * source.scale, p[1].get<double>() * source.scale,
                                          p[2].get<double>() * source.scale);
        if (fiber.points.size() >= 2)
            out.push_back(std::move(fiber));
    }
    std::stable_sort(out.begin(), out.end(),
                     [](const Fiber& a, const Fiber& b) { return a.points.size() > b.points.size(); });
    if (int(out.size()) > count)
        out.resize(std::size_t(count));
    if (out.empty())
        throw std::runtime_error("no fiber with line_points in " + source.path.string());
    return out;
}

VolumeSource Fixture::volumeSource(const fs::path& configDir)
{
    const auto& v = volume();
    auto levels = numbers("partial_levels");  // read in every state, so a case may set it
    VolumeSource source;
    if (cacheState_ == "cached") {
        source.location = v.path.string();
        return source;
    }
    if (v.url.empty())
        throw std::invalid_argument("cache state " + cacheState_ + ": volumes." + v.name +
                                    " has a local path and no url to fetch from");
    source.location = v.url;
    source.remote = true;
    if (prefetch_) {
        // Fetched chunks land in the cached copy timed runs open.
        source.cacheRoot = cacheRoot_;
    } else {
        if (!scratch_.empty())
            throw std::logic_error("Fixture::volumeSource called twice");
        const auto base = manifest_.contains("scratch")
                              ? resolveAgainst(configPath_.parent_path(), manifest_["scratch"].get<std::string>())
                              : fs::path(cacheRoot_.string() + "-scratch");
        scratch_ = fs::absolute(base) / (benchmark_ + "-" + caseName_ + "-" + std::to_string(::getpid()));
        fs::remove_all(scratch_);
        fs::create_directories(scratch_);
        source.cacheRoot = scratch_;
        if (cacheState_ == "partial") {
            // The scratch copy of the volume: its metadata and the seeded levels' chunks.
            const auto relative = fs::relative(v.path, cacheRoot_);
            const auto copy = scratch_ / relative;
            if (relative.empty() || *relative.begin() == "..")
                throw std::invalid_argument("cache state partial: volumes." + v.name + "'s copy " + v.path.string() +
                                            " is not inside the cache " + cacheRoot_.string());
            if (levels.empty())
                for (const auto& entry : fs::directory_iterator(v.path))
                    if (isLevelDir(entry) && entry.path().filename() != "0")
                        levels.push_back(std::stod(entry.path().filename().string()));
            std::sort(levels.begin(), levels.end());
            fs::create_directories(copy);
            std::size_t files = 0;
            std::string seeded;
            for (const auto& entry : fs::directory_iterator(v.path)) {
                const auto name = entry.path().filename().string();
                if (!isLevelDir(entry)) {
                    if (entry.is_regular_file()) {
                        fs::copy_file(entry.path(), copy / name);
                        ++files;
                    }
                    continue;
                }
                if (std::find(levels.begin(), levels.end(), std::stod(name)) != levels.end()) {
                    files += linkTree(entry.path(), copy / name);
                } else {
                    // An unseeded level keeps its metadata: the volume was opened before.
                    fs::create_directories(copy / name);
                    for (const auto& meta : fs::directory_iterator(entry.path()))
                        if (meta.is_regular_file() && meta.path().filename().string().front() == '.')
                            fs::copy_file(meta.path(), copy / name / meta.path().filename());
                }
            }
            for (const double level : levels)
                seeded += (seeded.empty() ? "" : ",") + std::to_string(int(level));
            config_["partial_levels"] = seeded;
            config_["partial_seeded_files"] = std::to_string(files);
        }
    }
    writeVc3dCacheConfig(configDir, source.cacheRoot);
    config_["volume_cache_root"] = source.cacheRoot.string();
    return source;
}

Report Fixture::makeReport() const
{
    return Report(benchmark_, pipeline_, caseName_, cacheState_, config_);
}

// ---------------------------------------------------------------------------
// Surfaces

namespace {
bool validVertex(const cv::Vec3f& p)
{
    return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]) && p[0] >= 0 && p[1] >= 0 &&
           p[2] >= 0;
}
} // namespace

std::shared_ptr<QuadSurface> loadSurface(const fs::path& tifxyz, float coordinateScale)
{
    auto loaded = load_quad_from_tifxyz(tifxyz);
    if (!loaded)
        throw std::runtime_error("cannot load " + tifxyz.string());
    cv::Mat_<cv::Vec3f> points = loaded->rawPoints().clone();
    for (auto& p : points)
        if (validVertex(p))
            p *= coordinateScale;
    return std::make_shared<QuadSurface>(points, loaded->scale() / coordinateScale);
}

cv::Vec3f surfaceCenter(QuadSurface& surface)
{
    const auto& points = *surface.rawPointsPtr();
    const int cy = points.rows / 2, cx = points.cols / 2;
    for (int r = 0; r < std::max(points.rows, points.cols); ++r)
        for (int y = std::max(0, cy - r); y <= std::min(points.rows - 1, cy + r); ++y)
            for (int x = std::max(0, cx - r); x <= std::min(points.cols - 1, cx + r); ++x)
                if (validVertex(points(y, x)))
                    return points(y, x);
    throw std::runtime_error("surface has no valid vertex");
}

std::vector<cv::Vec3f> surfaceSamplePoints(QuadSurface& surface, int count)
{
    const auto& points = *surface.rawPointsPtr();
    std::vector<cv::Vec3f> valid;
    // A coarse lattice keeps the choice deterministic and spread over the sheet.
    const int step = std::max(1, int(std::sqrt(double(points.total()) / std::max(1, count * 4))));
    for (int y = step / 2; y < points.rows; y += step)
        for (int x = step / 2; x < points.cols; x += step)
            if (validVertex(points(y, x)))
                valid.push_back(points(y, x));
    std::vector<cv::Vec3f> out;
    if (valid.empty())
        return out;
    for (int i = 0; i < count; ++i)
        out.push_back(valid[(std::size_t(i) * 7919) % valid.size()]);
    return out;
}

} // namespace vcbench
