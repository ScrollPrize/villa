// Zarr v2/v3 arrays and groups opened for reading, with their disk-cache layers.
#include "vc_reader/zarr.hpp"
#include "vc_reader/zarr_metadata.hpp"
#include "vc_reader/array.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <thread>
#include <set>

namespace fs = std::filesystem;

namespace vcr::zarr {
namespace {
std::string stripSlashes(std::string s)
{
    while (s.size() > 1 && s.back() == '/')
        s.pop_back();
    return s;
}

bool digits(const std::string& s)
{
    return !s.empty() && std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c); });
}

int levelNumber(const std::string& path)
{
    int value = 0;
    const auto [end, error] = std::from_chars(path.data(), path.data() + path.size(), value);
    if (error != std::errc{} || end != path.data() + path.size() || value < 0)
        throw std::invalid_argument("invalid numeric pyramid level: " + path);
    return value;
}

std::vector<std::string> localNumberedChildren(const fs::path& root)
{
    std::vector<std::string> paths;
    if (fs::is_directory(root))
        for (const auto& child : fs::directory_iterator(root)) {
            const auto name = child.path().filename().string();
            if (digits(name) && child.is_directory() &&
                (fs::exists(child.path() / ".zarray") || fs::exists(child.path() / "zarr.json")))
                paths.push_back(name);
        }
    return paths;
}

constexpr const char* kKeptGroup = ".vc_reader_group.json";

// The cache folders of a remote volume, where its metadata is kept.
std::vector<fs::path> keptFolders(const std::string& location, const OpenOptions& options)
{
    auto dirs = cachedNodeDirs(options.cacheDirs, options.access.cacheUrl(location));
    if (options.vc3dVolumeDir && fs::is_directory(*options.vc3dVolumeDir))
        dirs.insert(dirs.begin(), *options.vc3dVolumeDir);
    return dirs;
}

std::optional<Json> keptGroup(const std::string& location, const OpenOptions& options)
{
    if (options.offline || !isRemote(location) || !isPublicLocation(location))
        return std::nullopt;
    for (const auto& dir : keptFolders(location, options))
        if (std::ifstream in(dir / kKeptGroup); in)
            try {
                return Json::parse(in);
            } catch (const std::exception&) {
                // A damaged copy: read the store again.
            }
    return std::nullopt;
}

MetadataReader reader(const OpenOptions& options)
{
    return options.metadata ? options.metadata : metadataReader(options.access);
}

Json v3Attributes(const ArrayMeta& meta)
{
    if (!meta.v3)
        return Json::object();
    Json attrs = meta.v3->metadata.value("attributes", Json::object());
    if (meta.v3->metadata.contains("dimension_names") && !meta.v3->metadata.at("dimension_names").is_null())
        attrs["_ARRAY_DIMENSIONS"] = meta.v3->metadata.at("dimension_names");
    return attrs;
}

std::optional<Json> readJsonNode(const MetadataReader& read, const std::string& location, const std::string& name)
{
    const auto text = read(location, name);
    if (!text)
        return std::nullopt;
    return Json::parse(*text);
}

// x -> a * x + b on each spatial axis, composed from one OME dataset's transformations.
struct AxisMap {
    std::array<double, 3> a{1, 1, 1}, b{0, 0, 0};
    bool scaled = false, translated = false;
};

// Where ZYX of a placement come from in the OME axes, nullopt for a leading axis the
// array does not have (a 2-D image): (the spatial axes, or the last three of `rank` when
// axes are not declared.)
using SpatialAxes = std::array<std::optional<std::size_t>, 3>;
SpatialAxes spatialAxes(const Json& multiscale, std::size_t rank)
{
    std::vector<std::size_t> axes;
    const auto list = multiscale.value("axes", Json::array());
    for (std::size_t i = 0; i < list.size(); ++i) {
        const auto& e = list[i];
        const auto name = e.is_string() ? e.get<std::string>() : e.value("name", std::string());
        const auto type = e.is_object() ? e.value("type", std::string()) : std::string();
        if (type == "space" || (type.empty() && (name == "z" || name == "y" || name == "x")))
            axes.push_back(i);
    }
    if (list.empty())
        for (std::size_t i = rank > 3 ? rank - 3 : 0; i < rank; ++i)
            axes.push_back(i);
    if (axes.empty() || axes.size() > 3)
        throw std::invalid_argument("OME multiscales need one to three spatial axes to place pyramid levels");
    SpatialAxes out;
    std::copy(axes.begin(), axes.end(), out.begin() + std::ptrdiff_t(3 - axes.size()));
    return out;
}

AxisMap datasetMap(const Json& dataset, const SpatialAxes& axes)
{
    AxisMap m;
    for (const auto& t : dataset.value("coordinateTransformations", Json::array())) {
        const auto type = t.value("type", std::string());
        if (type == "identity")
            continue;
        if (type != "scale" && type != "translation")
            throw std::invalid_argument("unsupported OME dataset transformation '" + type + "'");
        const auto& v = t.at(type);
        for (int k = 0; k < 3; ++k) {
            if (!axes[k])
                continue;
            if (!v.is_array() || *axes[k] >= v.size() || !v[*axes[k]].is_number())
                throw std::invalid_argument("OME " + type + " does not cover the spatial axes");
            const double x = v[*axes[k]].get<double>();
            if (!std::isfinite(x) || (type == "scale" && !(x > 0)))
                throw std::invalid_argument("OME " + type + " values must be finite (and scales positive)");
            if (type == "scale") {
                m.a[k] *= x;
                m.b[k] *= x;
            } else {
                m.b[k] += x;
            }
        }
        (type == "scale" ? m.scaled : m.translated) = true;
    }
    return m;
}

// Level n of a pyramid whose levels each 2x2x2-reduce the one below.
LevelPlacement blockReduced(int n)
{
    if (n > 62)
        throw std::invalid_argument("pyramid level is too deep to place");
    const double s = std::ldexp(1.0, n);
    LevelPlacement p;
    p.scale = {s, s, s};
    p.offset = {(s - 1) / 2, (s - 1) / 2, (s - 1) / 2};
    return p;
}

std::string datasetPath(const Json& dataset)
{
    const auto& path = dataset.at("path");
    return path.is_string() ? path.get<std::string>() : path.dump();
}
}  // namespace

OpenedArray openArray(const std::string& input, const OpenOptions& options, ArrayHints hints)
{
    if (options.readMode != "mmap" && options.readMode != "slab")
        throw std::invalid_argument("read_mode must be 'mmap' or 'slab'");
    const auto location = stripSlashes(input);
    const bool remote = isRemote(location);
    const bool offline = options.offline && remote;
    if (offline && options.cacheDirs.empty() && !options.vc3dVolumeDir)
        throw std::invalid_argument("offline reads of a remote array need cache directories");
    if (options.cacheMaxBytes && *options.cacheMaxBytes < 0)
        throw std::invalid_argument("cache_max_bytes must be nonnegative");
    if (options.cacheMaxBytes && remote && options.cacheDirs.empty() && !options.vc3dVolumeDir)
        throw std::invalid_argument("cache_max_bytes requires cache directories");

    std::optional<std::string> group = hints.group, levelKey = hints.levelKey;
    if (group) {
        *group = stripSlashes(*group);
        if (!levelKey)
            levelKey = location.substr(group->size() + 1);
    } else if (!levelKey) {
        // An OME-Zarr level is a digits-only path under its group.
        const auto slash = location.rfind('/');
        if (slash != std::string::npos && slash > 0 && digits(location.substr(slash + 1))) {
            group = location.substr(0, slash);
            levelKey = location.substr(slash + 1);
        }
    }

    const auto& access = options.access;
    const auto keyUrl = remote ? access.cacheUrl(location) : location;
    const std::optional<std::string> groupUrl =
        remote && group ? std::optional<std::string>(access.cacheUrl(*group)) : std::nullopt;

    OpenedArray out;
    out.location = location;
    if (hints.metadata) {
        out.meta = parseArrayMeta(*hints.metadata);
    } else if (offline) {
        auto cached = cachedSourceZarray(options.cacheDirs, keyUrl, groupUrl, levelKey, options.vc3dVolumeDir);
        if (!cached)
            throw CacheMiss(location + " is not in the disk caches (offline)");
        out.meta = parseArrayMeta(*cached);
    } else {
        const auto read = reader(options);
        auto meta = readJsonNode(read, location, ".zarray");
        if (!meta) {
            meta = readJsonNode(read, location, "zarr.json");
            if (!meta || meta->value("node_type", "") != "array")
                throw NotFound("no Zarr array at " + location + " (.zarray/zarr.json not found)");
        }
        out.meta = parseArrayMeta(*meta);
    }
    out.store = group.value_or(location);
    out.levelKey = levelKey.value_or("");
    out.attributes = v3Attributes(out.meta);
    out.attributesRead = out.meta.v3.has_value();
    if (hints.attributes) {
        out.attributes = std::move(*hints.attributes);
        out.attributesRead = true;
    }
    // Offline, chunks only come from the caches: the source compressor never decodes anything.
    const std::string codec = offline ? std::string() : out.meta.codec();

    ArraySpec spec;
    spec.shape = out.meta.shape;
    spec.chunks = out.meta.chunks;
    spec.itemsize = out.meta.itemsize();
    spec.fill = out.meta.fill;
    spec.separator = out.meta.separator;
    spec.offline = offline;
    spec.read_mode = options.readMode;
    spec.http.connections = options.connections;
    std::string root;
    if (remote) {
        spec.remote_url = access.url(location);
        spec.http.auth = offline ? Auth{} : access.auth;
        if (!options.cacheDirs.empty() || options.vc3dVolumeDir) {
            LayerOptions layerOptions;
            layerOptions.groupUrl = groupUrl;
            layerOptions.levelKey = levelKey;
            layerOptions.vc3dVolumeDir = options.vc3dVolumeDir;
            layerOptions.writable = options.cacheWrites && !offline;
            layerOptions.allowLossy = options.allowLossyCache;
            layerOptions.compress = options.compressCache;
            layerOptions.maxBytes = options.cacheMaxBytes;
            auto layers = openLayers(options.cacheDirs, keyUrl, out.meta, layerOptions);
            out.layers = std::move(layers.layers);
            out.writer = layers.writer;
            out.busy = std::move(layers.busy);
        }
        if (offline && out.layers.empty())
            throw CacheMiss("no disk cache of " + location + " (offline)");
        for (const auto& layer : out.layers)
            spec.layers.push_back(layer.spec());
        spec.write_layer = out.writer;
        // A shared decoded cache must not let quantized reads satisfy an exact reader.
        spec.identity = Json::array({keyUrl, options.allowLossyCache}).dump();
    } else {
        std::error_code ec;
        root = fs::absolute(location, ec).lexically_normal().string();
        if (ec)
            root = location;
        while (root.size() > 1 && root.back() == '/')
            root.pop_back();
        if (!out.meta.v3)
            spec.layers.push_back(LayerSpec{"array", root, codec, "", false});
        spec.identity = "file://" + root;
    }
    if (remote || out.meta.v3)
        spec.remote_codec = codec;
    if (out.meta.v3) {
        spec.source_root = remote ? spec.remote_url : root;
        spec.source_prefix = out.meta.v3->prefix;
        spec.shard_chunks = out.meta.v3->shards;
        spec.shard_index_end = out.meta.v3->indexEnd;
        spec.shard_index_crc = out.meta.v3->indexCrc;
    }
    spec.threads = options.threads ? options.threads
                                   : remote ? 32 : std::min<std::size_t>(16, std::max(4u, std::thread::hardware_concurrency()));
    if (out.writer >= 0 && !out.layers[out.writer].budgetRoot.empty()) {
        const auto& writer=out.layers[out.writer];
        spec.disk_cache = diskBudget(writer.budgetRoot, options.cacheMaxBytes);
        if (options.cacheMinFreeBytes)
            spec.disk_cache->set_min_free_bytes(*options.cacheMinFreeBytes);
        if (writer.kind.starts_with("vc3d-"))
            spec.disk_cache->adopt_chunks(writer.budgetRoot);
    }
    out.array = std::make_unique<Array>(std::move(spec), options.memoryCache);
    return out;
}

OpenedArray::OpenedArray() = default;
OpenedArray::~OpenedArray() = default;
OpenedArray::OpenedArray(OpenedArray&&) noexcept = default;
OpenedArray& OpenedArray::operator=(OpenedArray&&) noexcept = default;

void OpenedArray::close()
{
    if (array)
        array->flush();
    for (auto& layer : layers)
        layer.lease.reset();
}

namespace {
constexpr const char* kSourceAttributes = ".vc_reader_source_attrs.json";

void keepAttributes(const OpenedArray& array)
{
    if (array.writer < 0)
        return;
    try {
        // The .zattrs of a decoded cache belongs to the cache, not the source.
        atomicWrite(fs::path(array.layers[std::size_t(array.writer)].root) / kSourceAttributes, array.attributes.dump());
    } catch (const std::exception&) {
        // A read-only cache: the attributes are still known to this reader.
    }
}
}  // namespace

const Json& arrayAttributes(OpenedArray& array, const OpenOptions& options)
{
    if (array.attributesRead)
        return array.attributes;
    if (options.offline && isRemote(array.location)) {
        auto dirs = cachedNodeDirs(options.cacheDirs, options.access.cacheUrl(array.location));
        if (options.vc3dVolumeDir && !array.levelKey.empty())
            dirs.insert(dirs.begin(), *options.vc3dVolumeDir / array.levelKey);
        for (const auto& dir : dirs) {
            std::ifstream in(dir / kSourceAttributes);
            if (in) {
                array.attributes = Json::parse(in);
                array.attributesRead = true;
                return array.attributes;
            }
        }
        throw CacheMiss("array attributes for " + array.location + " are not cached (offline)");
    }
    const auto text = reader(options)(array.location, ".zattrs");
    array.attributes = text ? Json::parse(*text) : Json::object();
    array.attributesRead = true;
    keepAttributes(array);
    return array.attributes;
}

// ---- groups ---------------------------------------------------------------------------

Group::Group(std::string location, OpenOptions options, Json attributes, int version)
    : location_(std::move(location)), options_(std::move(options)), attributes_(std::move(attributes)), version_(version)
{
    // One reader (and HTTP client) for every lookup in this group and the arrays opened through it.
    if (!options_.metadata)
        options_.metadata = metadataReader(options_.access);
}

Group Group::open(const std::string& input, OpenOptions options)
{
    const auto location = stripSlashes(input);
    const bool remote = isRemote(location);
    if (const auto kept = keptGroup(location, options))
        return fromState(*kept, std::move(options));
    if (options.offline && remote) {
        if (options.cacheDirs.empty() && !options.vc3dVolumeDir)
            throw std::invalid_argument("offline reads of a remote group need cache directories");
        auto dirs = cachedNodeDirs(options.cacheDirs, options.access.cacheUrl(location));
        if (options.vc3dVolumeDir && fs::is_directory(*options.vc3dVolumeDir))
            dirs.insert(dirs.begin(),*options.vc3dVolumeDir);
        if (dirs.empty())
            throw CacheMiss(location + " is not in the disk caches (offline)");
        Json attrs = Json::object();
        for (const auto& d : dirs) {
            std::ifstream in(d / ".zattrs");
            if (in) {
                attrs = Json::parse(in);
                break;
            }
        }
        return Group(location, std::move(options), std::move(attrs), 2);
    }
    if (!options.metadata)
        options.metadata = metadataReader(options.access);
    const auto read = options.metadata;
    auto zgroup = read(location, ".zgroup");
    auto zattrs = read(location, ".zattrs");
    int version = 2;
    if (!zgroup && !zattrs) {
        if (const auto node = readJsonNode(read, location, "zarr.json")) {
            if (node->value("zarr_format", 0) != 3 || node->value("node_type", "") != "group")
                throw std::invalid_argument(location + " is not a Zarr group");
            version = 3;
            // The disk layer stores decoded inner chunks as v2 arrays.
            zgroup = std::string(R"({"zarr_format": 2})");
            zattrs = node->value("attributes", Json::object()).dump();
        }
    }
    if (!zgroup && !zattrs) {
        if (read(location, ".zarray"))
            throw std::invalid_argument(location + " is an array, not a group");
        // Zarr v2 permits implicit parent groups in local hierarchies.
        if (remote || localNumberedChildren(location).empty())
            throw NotFound("no Zarr group at " + location + " (.zgroup/zarr.json not found)");
    }
    Json attrs = zattrs ? Json::parse(*zattrs) : Json::object();
    if (remote && options.cacheWrites) {
        const auto url=options.access.cacheUrl(location);
        auto targets=cachedNodeDirs(options.cacheDirs,url);
        if (options.vc3dVolumeDir) targets.push_back(*options.vc3dVolumeDir);
        std::optional<fs::path> decoded;
        const auto rootDir=std::find_if(options.cacheDirs.begin(),options.cacheDirs.end(),
                                        [](const std::string& d){return d!="vc3d";});
        if (rootDir!=options.cacheDirs.end()) {
            decoded=decodedCachePath(cacheRoots({*rootDir}).front(),url);
            if (std::find(targets.begin(),targets.end(),*decoded)==targets.end()) targets.push_back(*decoded);
        }
        for (const auto& target:targets) {
            try {
                // Mirror metadata obeys the same shared replacement lease as chunks.
                std::unique_ptr<Vc3dLease> lease;
                bool isDecoded=false;
                for (const auto& root:cacheRoots(options.cacheDirs))
                    isDecoded |= target==decodedCachePath(root,url);
                if (!isDecoded) {
                    lease=std::make_unique<Vc3dLease>(target);
                    if (!lease->acquire()) continue;
                }
                for (const auto& [name,data]:{std::pair{".zgroup",zgroup},std::pair{".zattrs",zattrs}}) {
                    if (!data) continue;
                    std::ifstream existing(target/name,std::ios::binary);
                    const std::string current((std::istreambuf_iterator<char>(existing)),{});
                    if (!existing || current!=*data) atomicWrite(target/name,*data);
                }
            } catch (const std::exception&) {
                // Read-only caches still allow online opens.
            }
        }
    }
    return Group(location, std::move(options), std::move(attrs), version);
}

Group Group::fromState(const Json& state, OpenOptions options)
{
    Group group(state.at("location").get<std::string>(), std::move(options), state.at("attributes"), state.at("version").get<int>());
    for (const auto& [key, node] : state.at("nodes").items()) {
        auto& entry = group.nodes_[key];
        if (!node.is_null())
            entry = {node.at(0).get<std::string>() == "array" ? NodeKind::Array : NodeKind::Group, node.at(1)};
    }
    if (const auto it = state.find("consolidated"); it != state.end())
        group.consolidated_ = it->is_null() ? std::optional<Json>{} : std::optional<Json>(*it);
    return group;
}

Json Group::state(bool inherited) const
{
    std::unique_lock lock(*mutex_, std::defer_lock);
    if (!inherited)
        lock.lock();
    Json nodes = Json::object();
    for (const auto& [key, node] : nodes_)
        nodes[key] = node ? Json::array({node->first == NodeKind::Array ? "array" : "group", node->second}) : Json();
    Json out{{"location", location_}, {"attributes", attributes_}, {"version", version_}, {"nodes", std::move(nodes)}};
    if (consolidated_)
        out["consolidated"] = consolidated_->value_or(Json());
    return out;
}

namespace {
Json multiscalesOf(const Json& attributes)
{
    Json ms = attributes.value("multiscales", Json());
    if (ms.is_null() && attributes.contains("ome"))
        ms = attributes.at("ome").value("multiscales", Json());
    return ms.is_array() ? ms : Json::array();
}

}  // namespace

Json Group::multiscales() const
{
    return multiscalesOf(attributes_);
}

const StoreMetadata& Group::metadata(bool refresh)
{
    std::lock_guard lock(*mutex_);
    if (!metadata_ || refresh)
        metadata_ = std::make_shared<const StoreMetadata>(volumeMetadata(location_, options_, refresh));
    return *metadata_;
}

std::vector<std::string> Group::keys()
{
    std::lock_guard lock(*mutex_);
    std::set<std::string> names;
    const auto meta = consolidated();
    const auto scan = [&](const fs::path& directory, bool skipHidden) {
        for (const auto& entry : fs::directory_iterator(directory)) {
            const auto name = entry.path().filename().string();
            if (entry.is_directory() && !(skipHidden && name.starts_with(".")) && isLocalZarrNode(entry.path()))
                names.insert(name);
        }
    };
    if (meta && meta->is_object()) {
        for (const auto& [key, value] : meta->items()) {
            const auto slash = key.find('/');
            if (slash == std::string::npos || key.find('/', slash + 1) != std::string::npos)
                continue;
            const auto leaf = key.substr(slash + 1);
            if (leaf == ".zarray" || leaf == ".zgroup" || leaf == ".zattrs")
                names.insert(key.substr(0, slash));
        }
    } else if (!isRemote(location_)) {
        scan(location_, false);
    } else if (options_.offline) {
        for (const auto& d : cachedNodeDirs(options_.cacheDirs, options_.access.cacheUrl(location_)))
            scan(d, true);
        if (options_.vc3dVolumeDir && fs::is_directory(*options_.vc3dVolumeDir))
            scan(*options_.vc3dVolumeDir, true);
    } else {
        for (const auto& name : listChildren(options_.access, location_))
            if (node(name))
                names.insert(name);
    }
    return {names.begin(), names.end()};
}

std::vector<std::string> Group::levels(int probeLevels)
{
    if (probeLevels < 0)
        throw std::invalid_argument("probeLevels must be nonnegative");
    std::lock_guard lock(*mutex_);
    std::vector<std::string> out;
    const auto ms = multiscales();
    if (!ms.empty()) {
        for (const auto& d : ms.at(0).value("datasets", Json::array())) {
            const auto& path = d.at("path");
            out.push_back(path.is_string() ? path.get<std::string>() : path.dump());
        }
        return out;
    }
    std::vector<std::string> candidates;
    if (!isRemote(location_)) {
        candidates = localNumberedChildren(location_);
    }
    const auto meta = consolidated();
    if (meta && meta->is_object()) {
        for (const auto& [key, value] : meta->items()) {
            const auto slash = key.find('/');
            const auto name = key.substr(0, slash);
            if (slash != std::string::npos && digits(name) && key.substr(slash) == "/.zarray")
                candidates.push_back(name);
        }
    }
    if (!isRemote(location_) || meta) {
        std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
            const int na = levelNumber(a), nb = levelNumber(b);
            return na == nb ? a < b : na < nb;
        });
        candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
        for (const auto& path : candidates)
            if (const auto n = node(path); n && n->first == NodeKind::Array)
                out.push_back(path);
        return out;
    }
    for (int level = 0; probeLevels == 0 || level < probeLevels; ++level) {
        const auto path = std::to_string(level);
        const auto n = node(path);
        if (n && n->first == NodeKind::Array)
            out.push_back(path);
        else if (probeLevels == 0)
            break;
    }
    return out;
}

LevelPlacement Group::placement(const std::string& path) const
{
    return levelPlacement(attributes_, path);
}

LevelPlacement levelPlacement(const Json& attributes, const std::string& path)
{
    const auto ms = multiscalesOf(attributes);
    if (ms.empty())
        return digits(path) ? blockReduced(levelNumber(path)) : LevelPlacement{};
    const auto datasets = ms.at(0).value("datasets", Json::array());
    std::size_t index = datasets.size();
    for (std::size_t i = 0; i < datasets.size(); ++i)
        if (datasetPath(datasets[i]) == path)
            index = i;
    if (index == datasets.size())
        return {};
    std::size_t rank = 3;
    for (const auto& t : datasets[0].value("coordinateTransformations", Json::array()))
        if (const auto type = t.value("type", std::string()); t.contains(type) && t.at(type).is_array())
            rank = t.at(type).size();
    const auto axes = spatialAxes(ms.at(0), rank);
    const auto ref = datasetMap(datasets[0], axes), level = datasetMap(datasets[index], axes);
    // OME before 0.4 declared no transformations: each dataset halves the previous.
    if (!ref.scaled && !level.scaled && !ref.translated && !level.translated)
        return blockReduced(int(index));
    LevelPlacement p;
    for (int k = 0; k < 3; ++k) {
        const double b = level.translated ? level.b[k] : ref.b[k] + (level.a[k] - ref.a[k]) / 2;
        p.scale[k] = level.a[k] / ref.a[k];
        p.offset[k] = (b - ref.b[k]) / ref.a[k];
    }
    return p;
}

namespace {

// A root array's metadata, when the location is one.
std::optional<Json> rootArrayMetadata(const std::string& location, const OpenOptions& options)
{
    if (options.offline && isRemote(location))
        return cachedSourceZarray(options.cacheDirs, options.access.cacheUrl(location), {}, {},
                                  options.vc3dVolumeDir);
    if (keptGroup(location, options))
        return std::nullopt;
    const auto read = reader(options);
    auto root = readJsonNode(read, location, ".zarray");
    if (!root) {
        const auto v3 = readJsonNode(read, location, "zarr.json");
        if (v3 && v3->value("node_type", "") == "array") root = v3;
    }
    return root;
}

// (level number, path) of a group's levels, in the group's order.
std::vector<std::pair<int, std::string>> numberedLevels(Group& group, int probeLevels)
{
    const auto paths = group.levels(probeLevels);
    const bool numeric = std::all_of(paths.begin(), paths.end(), digits);
    std::set<int> seen;
    std::vector<std::pair<int, std::string>> out;
    for (std::size_t i = 0; i < paths.size(); ++i) {
        const int number = numeric ? levelNumber(paths[i]) : static_cast<int>(i);
        if (!seen.insert(number).second)
            throw std::invalid_argument("duplicate pyramid level " + std::to_string(number));
        out.emplace_back(number, paths[i]);
    }
    return out;
}

}  // namespace

bool isLocalZarrNode(const std::filesystem::path& directory)
{
    std::error_code ec;
    for (const auto* name : {".zarray", ".zgroup", ".zattrs", "zarr.json"})
        if (std::filesystem::is_regular_file(directory / name, ec)) return true;
    return false;
}

bool isLocalZarrArray(const std::filesystem::path& directory)
{
    std::error_code ec;
    if (std::filesystem::exists(directory / ".zarray", ec)) return true;
    std::ifstream in(directory / "zarr.json");
    if (!in) return false;
    const auto node = Json::parse(in, nullptr, false);
    return node.is_object() && node.value("node_type", "") == "array";
}

// Keeps a pyramid's group state with its caches, every level looked up.
static void keepPyramid(Group& group, const std::vector<std::pair<int, std::string>>& levels)
{
    const auto& options = group.options();
    if (!options.cacheWrites || !isRemote(group.location()) || !isPublicLocation(group.location()) ||
        options.offline || keptGroup(group.location(), options))
        return;
    const auto dirs = keptFolders(group.location(), options);
    if (dirs.empty())
        return;
    for (const auto& [number, path] : levels)
        (void)group.node(path);
    const auto data = group.state().dump();
    for (const auto& dir : dirs)
        try {
            atomicWrite(dir / kKeptGroup, data);
            return;
        } catch (const std::exception&) {
            // Read-only: try the next.
        }
}

OpenedArray openPyramidLevel(const std::string& input, int level, const OpenOptions& options,
                             int probeLevels)
{
    if (level < 0 || probeLevels < 0)
        throw std::invalid_argument("pyramid level and probeLevels must be nonnegative");
    const auto location = stripSlashes(input);
    if (auto root = rootArrayMetadata(location, options)) {
        if (level != 0) throw std::out_of_range("a root array has only pyramid level 0");
        ArrayHints hints;
        hints.metadata = std::move(root);
        return openArray(location, options, std::move(hints));
    }
    auto group = Group::open(location, options);
    const auto levels = numberedLevels(group, probeLevels);
    keepPyramid(group, levels);
    for (const auto& [number, path] : levels)
        if (number == level) return group.array(path);
    throw std::out_of_range("pyramid level " + std::to_string(level) + " is not present in " + location);
}

std::vector<PyramidLevelInfo> pyramidLevels(const std::string& input, const OpenOptions& options, int probeLevels)
{
    if (probeLevels < 0)
        throw std::invalid_argument("probeLevels must be nonnegative");
    const auto location = stripSlashes(input);
    if (const auto root = rootArrayMetadata(location, options))
        return {{0, "", parseArrayMeta(*root)}};
    auto group = Group::open(location, options);
    std::vector<PyramidLevelInfo> out;
    const auto levels = numberedLevels(group, probeLevels);
    for (const auto& [number, path] : levels) {
        const auto node = group.node(path);
        if (!node || node->first != Group::NodeKind::Array)
            throw std::runtime_error("pyramid level " + path + " of " + location + " is not an array");
        out.push_back({number, path, parseArrayMeta(node->second)});
    }
    keepPyramid(group, levels);
    std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) { return a.level < b.level; });
    return out;
}

void forgetKeptMetadata(const std::string& location, const OpenOptions& options)
{
    forgetVolumeMetadata();
    for (const auto& dir : keptFolders(location, options)) {
        std::error_code ec;
        fs::remove(dir / kKeptGroup, ec);
        fs::remove(dir / ".vc_reader_metadata.json", ec);
    }
}

std::optional<Json> Group::consolidated()
{
    std::lock_guard lock(*mutex_);
    if (!consolidated_) {
        std::optional<Json> metadata;
        if (!(options_.offline && isRemote(location_)) && version_ == 2)
            if (const auto text = options_.metadata(location_, ".zmetadata"))
                metadata = Json::parse(*text).value("metadata", Json());
        consolidated_ = metadata;
    }
    return *consolidated_;
}

std::optional<std::pair<Group::NodeKind, Json>> Group::node(const std::string& input)
{
    std::string key = input;
    while (!key.empty() && key.front() == '/')
        key.erase(key.begin());
    while (!key.empty() && key.back() == '/')
        key.pop_back();
    if (key.empty())
        throw std::invalid_argument("invalid child path '" + input + "'");
    for (std::size_t b = 0; b <= key.size();) {
        const auto e = std::min(key.find('/', b), key.size());
        const auto part = key.substr(b, e - b);
        if (part.empty() || part == "." || part == "..")
            throw std::invalid_argument("invalid child path '" + input + "'");
        b = e + 1;
    }
    std::lock_guard lock(*mutex_);
    if (const auto it = nodes_.find(key); it != nodes_.end())
        return it->second;
    const auto location = location_ + "/" + key;
    std::optional<std::pair<NodeKind, Json>> found;
    const bool offline = options_.offline && isRemote(location_);
    if (const auto meta = consolidated()) {
        if (meta->contains(key + "/.zarray"))
            found = {NodeKind::Array, meta->at(key + "/.zarray")};
        else if (meta->contains(key + "/.zgroup") || meta->contains(key + "/.zattrs"))
            found = {NodeKind::Group, Json()};
    } else if (offline) {
        if (auto zarray = cachedSourceZarray(options_.cacheDirs, options_.access.cacheUrl(location),
                                             options_.access.cacheUrl(location_), key, options_.vc3dVolumeDir)) {
            found = {NodeKind::Array, *zarray};
        } else {
            for (const auto& d : cachedNodeDirs(options_.cacheDirs, options_.access.cacheUrl(location))) {
                std::error_code ec;
                if (fs::exists(d / ".zgroup", ec) || fs::exists(d / ".zattrs", ec)) {
                    found = {NodeKind::Group, Json()};
                    break;
                }
            }
        }
    } else if (version_ == 3) {
        if (const auto metadata = readJsonNode(options_.metadata, location, "zarr.json")) {
            const auto kind = metadata->value("node_type", "");
            if (kind != "array" && kind != "group")
                throw std::runtime_error(location + ": invalid Zarr v3 node_type");
            found = {kind == "array" ? NodeKind::Array : NodeKind::Group, kind == "array" ? *metadata : Json()};
        }
    } else {
        const auto read = options_.metadata;
        if (const auto zarray = readJsonNode(read, location, ".zarray"))
            found = {NodeKind::Array, *zarray};
        else if (read(location, ".zgroup"))
            found = {NodeKind::Group, Json()};
    }
    nodes_[key] = found;
    return found;
}

OpenedArray Group::array(const std::string& key)
{
    const auto n = node(key);
    if (!n || n->first != NodeKind::Array)
        throw std::out_of_range("no array '" + key + "' in " + location_);
    ArrayHints hints;
    hints.group = location_;
    hints.levelKey = key;
    hints.metadata = n->second;
    auto out = openArray(location_ + "/" + key, options_, std::move(hints));
    out.placement = placement(key);
    if (const auto meta = consolidated()) {
        out.attributes = meta->value(key + "/.zattrs", Json::object());
        out.attributesRead = true;
        keepAttributes(out);
    }
    return out;
}

Group Group::group(const std::string& key)
{
    const auto n = node(key);
    if (!n || n->first != NodeKind::Group)
        throw std::out_of_range("no group '" + key + "' in " + location_);
    const auto location = location_ + "/" + key;
    if (version_ == 3)
        return Group::open(location, options_);
    Json attrs = Json::object();
    if (options_.offline && isRemote(location_)) {
        for (const auto& d : cachedNodeDirs(options_.cacheDirs, options_.access.cacheUrl(location))) {
            std::ifstream in(d / ".zattrs");
            if (in) {
                attrs = Json::parse(in);
                break;
            }
        }
    } else if (const auto zattrs = options_.metadata(location, ".zattrs")) {
        attrs = Json::parse(*zattrs);
    }
    return Group(location, options_, std::move(attrs), version_);
}

}  // namespace vcr::zarr
