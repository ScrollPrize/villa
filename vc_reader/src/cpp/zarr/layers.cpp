// Finding the on-disk chunk caches of a remote array, in the formats other tools
// already write, and choosing the one fetched chunks are appended to.
//
// * decoded (vc_reader, vesuvius RemoteChunkedArray): <root>/<scheme>/<host or
//   bucket>/<path>/ is a zarr v2 array of the decoded chunks, uncompressed or
//   losslessly Delta3D-encoded. vc_reader marks missing chunks <key>.missing.
// * vc3d-mirror (VC3D ZarrMirror): <root>/<name>-<fnv1a64(url)>/ mirrors the
//   source group; chunks are the source's stored bytes, <key>.empty marks a
//   missing chunk.
// * vc3d-delta3d (VC3D Delta3d): the same directory with a .vc_delta3d_cache
//   marker; chunks are decoded voxels re-encoded with lossless VC-Delta3D and
//   missing chunks are marked under .vc_cache_empty/<level>/<key>.empty.
//
// VC3D caches are used under VC3D's shared lease (flock on
// <root>/.<id>.vc_cache.lock): VC3D replaces a cache directory only when it can
// take that lock exclusively.
#include "vc_reader/catalog.hpp"
#include "vc_reader/zarr.hpp"
#include "vc_reader/array.hpp"
#include "files.hpp"

#include <cmath>
#include <fstream>
#include <mutex>
#include <sstream>
#include <thread>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace vcr::zarr {
namespace {
constexpr const char* kDelta3d = "vc-delta3d";
constexpr const char* kDelta3dMarker = ".vc_delta3d_cache";
constexpr const char* kDelta3dMarkerContents = "D3D1\n";
constexpr const char* kVc3d = "vc3d";

std::optional<Json> readJson(const fs::path& path)
{
    std::ifstream in(path);
    if (!in)
        return std::nullopt;
    try {
        return Json::parse(in);
    } catch (const Json::exception&) {
        return std::nullopt;
    }
}

std::optional<std::string> readText(const fs::path& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        return std::nullopt;
    std::ostringstream text;
    text << in.rdbuf();
    return text.str();
}

// JSON equality that treats NaN fill values as equal.
bool metadataEqual(const Json& a, const Json& b)
{
    if (a.is_object() && b.is_object()) {
        if (a.size() != b.size())
            return false;
        for (const auto& [key, value] : a.items())
            if (!b.contains(key) || !metadataEqual(value, b.at(key)))
                return false;
        return true;
    }
    if (a.is_array() && b.is_array()) {
        if (a.size() != b.size())
            return false;
        for (std::size_t i = 0; i < a.size(); ++i)
            if (!metadataEqual(a[i], b[i]))
                return false;
        return true;
    }
    if (a.is_number_float() && b.is_number_float() && std::isnan(a.get<double>()) && std::isnan(b.get<double>()))
        return true;
    return a == b;
}

bool losslessCodec(const Json& config)
{
    if (!config.is_object() || config.value("id", std::string()) != kDelta3d)
        return true;
    const auto quant = config.value("quant", Json(1));
    return !quant.is_number() || quant.get<double>() <= 1;
}

// Whether a cache's .zarray describes the same array as the source (and, unless
// ignoreCodec, stores it with the same compressor, filters and order).
bool sameArray(const Json& stored, const ArrayMeta& source, bool ignoreCodec)
{
    ArrayMeta s;
    try {
        s = parseArrayMeta(stored);
    } catch (const std::exception&) {
        return false;
    }
    const auto nativeFill = [](const ArrayMeta& m) {
        auto f = m.fill;
        if (m.type.swap)
            std::reverse(f.begin(), f.end());
        return f;
    };
    bool sameFill;
    if (source.type.kind == 'f') {
        const double a = s.type.load(s.fill.data()), b = source.type.load(source.fill.data());
        sameFill = a == b || (std::isnan(a) && std::isnan(b));
    } else {
        sameFill = nativeFill(s) == nativeFill(source);
    }
    const bool same = s.type.kind == source.type.kind && s.type.itemsize == source.type.itemsize &&
                      s.shape == source.shape && s.chunks == source.chunks && s.separator == source.separator && sameFill;
    if (!same || ignoreCodec)
        return same;
    const auto noFilters = [](const Json& f) { return f.is_null() || f.empty(); };
    return stored.value("compressor", Json()) == source.compressor &&
           (noFilters(stored.value("filters", Json())) ? noFilters(source.filters)
                                                       : stored.value("filters", Json()) == source.filters) &&
           s.order == source.order;
}

std::uint64_t fnv1a64(const std::string& text)
{
    std::uint64_t h = 14695981039346656037ull;
    for (const unsigned char b : text)
        h = (h ^ b) * 1099511628211ull;
    return h;
}

std::vector<std::string> urlSpellings(const std::string& url)
{
    std::vector<std::string> out;
    auto add = [&](std::string u) {
        while (!u.empty() && u.back() == '/')
            u.pop_back();
        if (!u.empty() && std::find(out.begin(), out.end(), u) == out.end())
            out.push_back(u);
    };
    add(url);
    if (url.starts_with("http") || url.starts_with("s3"))
        add(catalog::resolveOpenDataUrl(url));
    if (auto s3 = catalog::openDataS3Url(url))
        add(*s3);
    add(httpUrl(url));
    return out;
}

std::optional<fs::path> vc3dDir(const fs::path& root, const std::string& groupUrl)
{
    for (const auto& spelling : urlSpellings(groupUrl)) {
        const auto dir = root / vc3dVolumeId(spelling);
        std::error_code ec;
        if (fs::is_directory(dir, ec))
            return dir;
    }
    return std::nullopt;
}

std::optional<CacheLayer> decodedLayer(const fs::path& root, const ArrayMeta& source, const std::string& url,
                                       bool create, bool compress = false, bool allowLossy = false)
{
    auto stored = readJson(root / ".zarray");
    if (!stored) {
        if (!create)
            return std::nullopt;
        if (compress && (source.shape.size() != 3 || source.type.kind != 'u' ||
                         (source.type.itemsize != 1 && source.type.itemsize != 2)))
            throw std::invalid_argument("compress_cache needs 3-D uint8 or uint16 chunks");
        stored = compress ? source.delta3dZarray() : source.decodedZarray();
        const Json record{{"vc_reader", {{"source_url", url}, {"source_zarray", source.raw}}}};
        atomicWrite(root / ".zattrs", record.dump(1));
        atomicWrite(root / ".zarray", stored->dump(1));
        CacheLayer layer;
        layer.kind = "decoded";
        layer.root = root.string();
        layer.stored = parseArrayMeta(*stored);
        return layer;
    }
    const auto attrs = readJson(root / ".zattrs");
    bool ok;
    if (attrs && attrs->contains("vc_reader")) {  // ours: uncompressed or Delta3D, as its .zarray says
        const auto& record = attrs->at("vc_reader");
        const auto comp = stored->value("compressor", Json());
        const auto filters = stored->value("filters", Json());
        ok = metadataEqual(record.value("source_zarray", Json()), source.raw) && (filters.is_null() || filters.empty()) &&
             (comp.is_null() || comp.value("id", std::string()) == kDelta3d);
        if (ok && !allowLossy && !losslessCodec(comp))
            return std::nullopt;
    } else {  // written by another tool (vesuvius): usable if it holds the same decoded array
        const auto filters = stored->value("filters", Json());
        const auto order = stored->value("order", Json());
        ok = stored->value("compressor", Json()).is_null() && (filters.is_null() || filters.empty()) &&
             (order.is_null() || order == "C") && sameArray(*stored, source, true);
    }
    if (!ok)
        throw std::invalid_argument("disk cache " + root.string() + " holds a different array than " + url +
                                 " (the remote .zarray changed?); delete that cache directory to refill it");
    CacheLayer layer;
    layer.kind = "decoded";
    layer.root = root.string();
    layer.stored = parseArrayMeta(*stored);
    layer.allowLossy = allowLossy;
    return layer;
}

std::optional<CacheLayer> vc3dLayer(const fs::path& volumeDir, const std::string& levelKey, const ArrayMeta& source,
                                    bool allowLossy, bool writable, std::vector<std::string>& busy)
{
    if (source.v3)
        return std::nullopt;  // VC3D mirror layouts describe v2 source chunks
    auto lease = std::make_shared<Vc3dLease>(volumeDir);
    if (!lease->acquire()) {  // being replaced by another process
        busy.push_back(volumeDir.string());
        return std::nullopt;
    }
    const bool delta3d = readText(volumeDir / kDelta3dMarker) == std::string(kDelta3dMarkerContents);
    const auto root = fs::weakly_canonical(volumeDir / levelKey);
    auto stored = readJson(root / ".zarray");
    if (!stored && writable) {
        // VC3D publishes each level's metadata when it first touches the level; do the same.
        stored = delta3d ? source.delta3dZarray() : source.raw;
        atomicWrite(root / ".zarray", stored->dump(2) + "\n");
    }
    if (!stored)
        return std::nullopt;
    CacheLayer layer;
    layer.root = root.string();
    layer.lease = lease;
    if (delta3d) {
        const auto comp = stored->value("compressor", Json());
        if (comp.value("id", std::string()) != kDelta3d || !sameArray(*stored, source, true))
            throw std::invalid_argument("VC3D Delta3D cache " + root.string() + " does not hold this array's chunks");
        if (!allowLossy && !losslessCodec(comp))
            return std::nullopt;
        layer.kind = "vc3d-delta3d";
        layer.stored = parseArrayMeta(*stored);
        layer.emptyRoot = (volumeDir / ".vc_cache_empty" / levelKey).string();
        layer.allowLossy = allowLossy;
        return layer;
    }
    if (!sameArray(*stored, source, false))
        throw std::invalid_argument("VC3D cache " + root.string() + " mirrors a different array (the remote .zarray changed?)");
    layer.kind = "vc3d-mirror";
    layer.stored = parseArrayMeta(*stored);
    return layer;
}
}  // namespace

std::string vc3dVolumeId(std::string url)
{
    while (!url.empty() && url.back() == '/')
        url.pop_back();
    auto name = url.substr(url.rfind('/') + 1);
    if (name.empty())
        name = "remote";
    char hex[17];
    std::snprintf(hex, sizeof hex, "%016llx", static_cast<unsigned long long>(fnv1a64(url)));
    return name + "-" + hex;
}

void createVc3dVolumeDir(const fs::path& volumeDir, bool delta3d)
{
    std::error_code ec;
    if (fs::is_directory(volumeDir, ec))
        return;
    fs::create_directories(volumeDir);
    if (delta3d)
        atomicWrite(volumeDir / kDelta3dMarker, kDelta3dMarkerContents);
}

std::optional<fs::path> vc3dCacheRoot()
{
    const char* configured = std::getenv("VC3D_CONFIG_DIR");
    const char* home = std::getenv("HOME");
    const fs::path configDir = configured && *configured ? fs::path(configured) : fs::path(home ? home : "") / ".VC3D";
    std::vector<fs::path> candidates;
    std::ifstream ini(configDir / "VC3D.ini");
    std::string line, section;
    while (std::getline(ini, line)) {
        const auto b = line.find_first_not_of(" \t\r");
        if (b == std::string::npos)
            continue;
        line = line.substr(b, line.find_last_not_of(" \t\r") - b + 1);
        if (line.front() == '[' && line.back() == ']') {
            section = line.substr(1, line.size() - 2);
        } else if (const auto eq = line.find('='); section == "viewer" && eq != std::string::npos) {
            auto key = line.substr(0, eq);
            key.erase(key.find_last_not_of(" \t") + 1);
            auto value = line.substr(eq + 1);
            value.erase(0, value.find_first_not_of(" \t\""));
            value.erase(value.find_last_not_of(" \t\"") + 1);
            if (key == "remote_cache_dir" && !value.empty()) {
                if (value.starts_with("~/") && home)
                    value = std::string(home) + value.substr(1);
                candidates.emplace_back(value);
            }
        }
    }
    for (const char* base : {"/volpkgs", "/ephemeral"}) {
        std::error_code ec;
        if (fs::is_directory(base, ec)) {
            candidates.push_back(fs::path(base) / "remote_cache");
            break;
        }
    }
    candidates.push_back(fs::path(home ? home : "") / ".VC3D" / "remote_cache");
    for (const auto& c : candidates) {
        std::error_code ec;
        if (fs::is_directory(c, ec))
            return c;
    }
    return std::nullopt;
}

fs::path decodedCachePath(const fs::path& cacheDir, const std::string& input)
{
    auto url = catalog::openDataS3Url(input).value_or(input);
    while (!url.empty() && url.back() == '/')
        url.pop_back();
    const auto scheme = url.find("://");
    if (scheme == std::string::npos)
        throw std::invalid_argument("not a URL: '" + url + "'");
    auto rest = url.substr(scheme + 3);
    const auto query = rest.find_first_of("?#");
    if (query != std::string::npos)
        rest.erase(query);
    const auto slash = rest.find('/');
    auto out = cacheDir / url.substr(0, scheme) / rest.substr(0, slash);
    if (slash != std::string::npos)
        out /= rest.substr(slash + 1);
    return out;
}

std::shared_ptr<DiskCache> diskBudget(const std::string& root, std::optional<std::int64_t> maxBytes)
{
    static std::mutex mutex;
    static std::map<std::string, std::shared_ptr<DiskCache>> budgets;
    static pid_t owner = getpid();
    std::lock_guard lock(mutex);
    if (owner != getpid()) {  // a forked child: the parent's indexes are not ours to use
        for (auto& [_, budget] : budgets)
            new std::shared_ptr<DiskCache>(std::move(budget));  // leaked, never touched
        budgets.clear();
        owner = getpid();
    }
    auto& budget = budgets[root];
    if (!budget)
        budget = std::make_shared<DiskCache>(root, maxBytes.value_or(-1));
    else if (maxBytes)
        budget->set_max_bytes(*maxBytes);
    return budget;
}

void atomicWrite(const fs::path& path, const std::string& data)
{
    detail::atomicWriteFile(path, [&](std::ostream& out) { out.write(data.data(), std::streamsize(data.size())); });
}

void detail::atomicWriteFile(const fs::path& path, const std::function<void(std::ostream&)>& write)
{
    fs::create_directories(path.parent_path());
    std::ostringstream suffix;
    suffix << ".tmp." << getpid() << "." << std::this_thread::get_id();
    const auto tmp = path.string() + suffix.str();
    try {
        {
            std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
            write(out);
            out.close();
            if (!out)
                throw std::runtime_error("cannot write " + tmp);
        }
        fs::rename(tmp, path);
    } catch (...) {
        std::error_code ec;
        fs::remove(tmp, ec);
        throw;
    }
}

Vc3dLease::Vc3dLease(const fs::path& volumeDir)
    : path_(volumeDir.parent_path() / ("." + volumeDir.filename().string() + ".vc_cache.lock"))
{
}

Vc3dLease::~Vc3dLease() { release(); }

bool Vc3dLease::acquire()
{
#ifdef _WIN32
    return false;  // no flock: VC3D caches are not shared
#else
    const int fd = ::open(path_.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0666);
    if (fd < 0)
        return false;
    if (::flock(fd, LOCK_SH | LOCK_NB) != 0) {
        ::close(fd);
        return false;
    }
    fd_ = fd;
    return true;
#endif
}

void Vc3dLease::release()
{
#ifndef _WIN32
    if (fd_ >= 0)
        ::close(fd_);  // closing the descriptor drops the flock
#endif
    fd_ = -1;
}

LayerSpec CacheLayer::spec() const
{
    return LayerSpec{kind, root, stored.codec(), emptyRoot, allowLossy};
}

std::vector<fs::path> cacheRoots(const std::vector<std::string>& cacheDirs)
{
    std::vector<fs::path> roots;
    for (const auto& d : cacheDirs) {
        if (d == kVc3d) {
            if (auto r = vc3dCacheRoot())
                roots.push_back(*r);
        } else {
            const char* home = std::getenv("HOME");
            roots.push_back(d.starts_with("~/") && home ? fs::path(home) / d.substr(2) : fs::path(d));
        }
    }
    return roots;
}

std::optional<Json> cachedSourceZarray(const std::vector<std::string>& cacheDirs, const std::string& url,
                                       const std::optional<std::string>& groupUrl,
                                       const std::optional<std::string>& levelKey,
                                       const std::optional<fs::path>& vc3dVolumeDir)
{
    std::optional<Json> fallback;
    auto fromVc3d = [&](const fs::path& vdir) -> std::optional<Json> {
        auto stored = readJson(vdir / *levelKey / ".zarray");
        if (stored) {
            std::error_code ec;
            if (!fs::exists(vdir / kDelta3dMarker, ec))
                return stored;
            if (!fallback)
                fallback = stored;
        }
        return std::nullopt;
    };
    if (vc3dVolumeDir && levelKey)
        if (auto exact = fromVc3d(*vc3dVolumeDir))
            return exact;
    for (const auto& root : cacheRoots(cacheDirs)) {
        if (groupUrl && levelKey && !vc3dVolumeDir)
            if (const auto vdir = vc3dDir(root, *groupUrl))
                if (auto exact = fromVc3d(*vdir))
                    return exact;
        const auto droot = decodedCachePath(root, url);
        const auto attrs = readJson(droot / ".zattrs");
        if (attrs && attrs->contains("vc_reader")) {
            const auto source = attrs->at("vc_reader").value("source_zarray", Json());
            if (!source.is_null() && !source.empty())
                return std::optional<Json>{source};
        }
        if (auto stored = readJson(droot / ".zarray"); stored && !fallback)
            fallback = stored;
    }
    return fallback;
}

std::vector<fs::path> cachedNodeDirs(const std::vector<std::string>& cacheDirs, const std::string& url)
{
    std::vector<fs::path> out;
    for (const auto& root : cacheRoots(cacheDirs)) {
        std::error_code ec;
        if (const auto d = decodedCachePath(root, url); fs::is_directory(d, ec))
            out.push_back(d);
        std::string parent = url;
        while (!parent.empty() && parent.back() == '/')
            parent.pop_back();
        std::vector<std::string> rel;
        while (true) {
            const auto scheme = parent.find("://");
            const auto after = scheme == std::string::npos ? 0 : scheme + 3;
            if (parent.find('/', after) == std::string::npos)
                break;
            if (const auto vdir = vc3dDir(root, parent)) {
                fs::path d = *vdir;
                for (auto it = rel.rbegin(); it != rel.rend(); ++it)
                    d /= *it;
                if (fs::is_directory(d, ec))
                    out.push_back(d);
                break;
            }
            const auto slash = parent.rfind('/');
            rel.push_back(parent.substr(slash + 1));
            parent.erase(slash);
        }
    }
    return out;
}

Layers openLayers(const std::vector<std::string>& cacheDirs, const std::string& url, const ArrayMeta& source,
                  const LayerOptions& options)
{
    std::vector<fs::path> roots;
    for (const auto& r : cacheRoots(cacheDirs)) {
        std::error_code ec;
        const auto canonical = fs::weakly_canonical(r, ec);
        roots.push_back(ec ? r : canonical);
    }
    Layers out;
    const bool vc3dWritable = options.writable;
    if (options.vc3dVolumeDir && options.levelKey) {
        std::error_code ec;
        if (fs::is_directory(*options.vc3dVolumeDir, ec))
            if (auto layer = vc3dLayer(*options.vc3dVolumeDir, *options.levelKey, source, options.allowLossy, vc3dWritable, out.busy))
                out.layers.push_back(std::move(*layer));
    }
    for (const auto& root : roots) {
        if (options.groupUrl && options.levelKey && !options.vc3dVolumeDir)
            if (const auto vdir = vc3dDir(root, *options.groupUrl))
                if (auto layer = vc3dLayer(*vdir, *options.levelKey, source, options.allowLossy, vc3dWritable, out.busy))
                    out.layers.push_back(std::move(*layer));
        if (auto layer = decodedLayer(decodedCachePath(root, url), source, url, false, false, options.allowLossy))
            out.layers.push_back(std::move(*layer));
    }
    if (!options.writable)
        return out;
    // Keep writing the selected VC3D format when a byte limit is configured.
    // Its budget spans the volume directories under the same cache root.
    if (!out.layers.empty() && out.layers.front().kind.starts_with("vc3d-")) {
        const auto root=fs::path(out.layers.front().root).parent_path().parent_path();
        std::error_code ec;
        if (options.maxBytes || fs::is_regular_file(root / ".vc_reader_cache.sqlite3",ec))
            out.layers.front().budgetRoot=root.string();
        out.writer=0;
        return out;
    }
    std::optional<fs::path> configured;
    for (const auto& root : roots) {
        std::error_code ec;
        if (fs::is_regular_file(root / ".vc_reader_cache.sqlite3", ec)) {
            configured = root;
            break;
        }
    }
    if (options.maxBytes || configured) {
        std::optional<fs::path> root = configured;
        if (options.maxBytes) {
            root.reset();
            for (const auto& d : cacheDirs)
                if (d != kVc3d) {
                    std::error_code ec;
                    const auto r = cacheRoots({d}).front();
                    const auto canonical = fs::weakly_canonical(r, ec);
                    root = ec ? r : canonical;
                    break;
                }
            if (!root)
                throw std::invalid_argument("cache_max_bytes needs an explicit decoded cache directory (not only 'vc3d')");
        }
        const auto target = decodedCachePath(*root, url).string();
        auto writer = std::find_if(out.layers.begin(), out.layers.end(),
                                   [&](const CacheLayer& l) { return l.kind == "decoded" && l.root == target; });
        if (writer == out.layers.end()) {
            auto layer = decodedLayer(target, source, url, true, options.compress, options.allowLossy);
            if (!layer)
                throw std::invalid_argument("bounded cache " + target + " is lossy; use a different cache directory");
            out.layers.push_back(std::move(*layer));
            writer = std::prev(out.layers.end());
        }
        writer->budgetRoot = root->string();
        out.writer = int(writer - out.layers.begin());
        return out;
    }
    if (!out.layers.empty()) {
        out.writer = 0;
        return out;
    }
    if (roots.empty())
        return out;
    out.layers.push_back(*decodedLayer(decodedCachePath(roots.front(), url), source, url, true, options.compress));
    out.writer = 0;
    return out;
}

}  // namespace vcr::zarr
