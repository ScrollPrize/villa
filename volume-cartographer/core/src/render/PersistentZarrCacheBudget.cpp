#include "vc/core/render/PersistentZarrCacheBudget.hpp"

#include "vc/core/render/ChunkFetch.hpp"

#include "vc/core/util/Logging.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

namespace vc::render {
namespace fs = std::filesystem;
namespace {

constexpr std::string_view kRetiredCacheDirectory = ".vc_cache_retired";

fs::path normalizedPath(const fs::path& path)
{
    std::error_code ec;
    auto result = fs::weakly_canonical(path, ec);
    if (ec) {
        ec.clear();
        result = fs::absolute(path, ec);
    }
    return (ec ? path : result).lexically_normal();
}

bool isWithin(const fs::path& path, const fs::path& root)
{
    auto p = path.begin();
    auto r = root.begin();
    for (; r != root.end(); ++r, ++p) {
        if (p == path.end() || *p != *r)
            return false;
    }
    return true;
}

fs::path retiredCachePath(const fs::path& subtree)
{
    static const auto processTag =
        static_cast<std::uint64_t>(std::random_device{}());
    static std::atomic<std::uint64_t> counter{0};
    return subtree.parent_path() / kRetiredCacheDirectory /
           (subtree.filename().string() + "." + std::to_string(processTag) +
            "." + std::to_string(
                      counter.fetch_add(1, std::memory_order_relaxed)));
}

bool isUnsignedNumber(const std::string& value)
{
    return !value.empty() &&
           std::all_of(value.begin(), value.end(), [](unsigned char c) {
               return c >= '0' && c <= '9';
           });
}

bool isVolumeChunk(const fs::path& path, const fs::path& root)
{
    const auto ext = path.extension().string();
    if (ext != ".bin" && ext != ".zst" && ext != ".c3d" &&
        ext != kPersistentSourcePayloadExtension && ext != ".empty")
        return false;
    auto y = path.parent_path();
    auto z = y.parent_path();
    auto level = z.parent_path();
    const auto relative = path.lexically_relative(root);
    if (relative.empty() ||
        (relative.begin() != relative.end() && *relative.begin() == ".."))
        return false;
    for (const auto& component : relative) {
        const auto value = component.string();
        if (value == "segments" || value == "normal_grids" ||
            value == "normal-grid" || value == "projects")
            return false;
    }
    const auto levelName = level.filename().string();
    return isUnsignedNumber(y.filename().string()) &&
           isUnsignedNumber(z.filename().string()) &&
           levelName.rfind("level_", 0) == 0 &&
           isUnsignedNumber(levelName.substr(6));
}

bool isZarrMetadataFile(const fs::path& path)
{
    const auto name = path.filename().string();
    return name == ".zarray" || name == ".zattrs" || name == ".zgroup" ||
           name == ".zmetadata" || name == "zarr.json";
}

struct NativeZarrArrayLayout {
    fs::path root;
    std::size_t rank = 0;
    std::string separator;
    bool chunkPrefix = false;
};

std::optional<NativeZarrArrayLayout>
nativeZarrArrayLayout(const fs::path& path)
{
    try {
        std::ifstream input(path);
        const auto metadata = nlohmann::json::parse(input);
        NativeZarrArrayLayout result;
        result.root = path.parent_path();
        if (path.filename() == ".zarray") {
            if (metadata.value("zarr_format", 0) != 2 ||
                !metadata.contains("shape") ||
                !metadata["shape"].is_array()) {
                return std::nullopt;
            }
            result.rank = metadata["shape"].size();
            result.separator = metadata.value("dimension_separator", ".");
        } else if (path.filename() == "zarr.json" &&
                   metadata.value("zarr_format", 0) == 3 &&
                   metadata.value("node_type", std::string{}) == "array" &&
                   metadata.contains("shape") &&
                   metadata["shape"].is_array()) {
            result.rank = metadata["shape"].size();
            const auto encoding = metadata.value(
                "chunk_key_encoding", nlohmann::json::object());
            const auto name = encoding.value("name", "default");
            const auto configuration = encoding.value(
                "configuration", nlohmann::json::object());
            if (name == "default") {
                result.chunkPrefix = true;
                result.separator = configuration.value("separator", "/");
            } else if (name == "v2") {
                result.separator = configuration.value("separator", ".");
            } else {
                return std::nullopt;
            }
        } else {
            return std::nullopt;
        }
        if (result.rank == 0 ||
            (result.separator != "." && result.separator != "/")) {
            return std::nullopt;
        }
        return result;
    } catch (...) {
        return std::nullopt;
    }
}

bool hasNumericComponents(std::string_view key,
                          std::string_view separator,
                          std::size_t rank)
{
    std::size_t count = 0;
    while (true) {
        const auto end = key.find(separator);
        const auto component = key.substr(0, end);
        if (!isUnsignedNumber(std::string(component)))
            return false;
        ++count;
        if (end == std::string_view::npos)
            return count == rank;
        key.remove_prefix(end + separator.size());
    }
}

bool isNativeZarrPayload(const fs::path& path,
                         const std::vector<NativeZarrArrayLayout>& arrays)
{
    if (isZarrMetadataFile(path) || path.filename().string().ends_with(".tmp"))
        return false;
    return std::any_of(arrays.begin(), arrays.end(), [&](const auto& array) {
        if (!isWithin(path, array.root) || path == array.root)
            return false;
        auto key = path.lexically_relative(array.root).generic_string();
        if (key.ends_with(".empty"))
            key.resize(key.size() - std::string_view(".empty").size());
        if (array.chunkPrefix) {
            const auto prefix = std::string("c") + array.separator;
            if (!key.starts_with(prefix))
                return false;
            key.erase(0, prefix.size());
        }
        return hasNumericComponents(key, array.separator, array.rank);
    });
}

bool isLasagnaData(const fs::path& path, const std::vector<fs::path>& artifacts)
{
    for (const auto& artifact : artifacts) {
        if (!isWithin(path, artifact))
            continue;
        const auto relative = path.lexically_relative(artifact);
        if (relative.empty())
            return false;
        for (const auto& component : relative) {
            if (component == ".lasagna-zarr-metadata")
                return false;
        }
        const auto name = path.filename().string();
        if (name.find(".tmp") != std::string::npos ||
            name == "lasagna-remote.json" || name == "zarr.json" ||
            name == ".zarray" || name == ".zattrs" || name == ".zgroup" ||
            path.extension() == ".json") {
            return false;
        }
        return true;
    }
    return false;
}

std::set<std::string> managedGenericPayloads(const std::vector<fs::path>& files)
{
    constexpr std::string_view suffix = ".vc-remote-file.json";
    std::set<std::string> payloads;
    for (const auto& sidecar : files) {
        const auto text = sidecar.string();
        if (!text.ends_with(suffix))
            continue;
        try {
            std::ifstream input(sidecar);
            const auto metadata = nlohmann::json::parse(input);
            if (metadata.value("accounting", std::string{}) != "managed")
                continue;
            const auto payload = fs::path(text.substr(0, text.size() - suffix.size()));
            std::error_code ec;
            if (fs::is_regular_file(payload, ec) && !ec)
                payloads.insert(normalizedPath(payload).string());
        } catch (...) {
        }
    }
    return payloads;
}

struct Registry {
    std::mutex mutex;
    std::map<fs::path, std::shared_ptr<PersistentZarrCacheBudget>> budgets;
};

Registry& registry()
{
#if defined(_WIN32)
    static auto* value = new Registry;
    return *value;
#else
    static Registry value;
    return value;
#endif
}

} // namespace

struct PersistentZarrCacheBudget::Impl {
    struct Entry {
        std::uint64_t size = 0;
        fs::file_time_type touched{};
    };

    fs::path root;
    Limits limits;
    SpaceProvider spaceProvider;
    std::mutex mutex;
    std::condition_variable cv;
    std::unordered_map<std::string, Entry> entries;
    std::unordered_map<std::string, std::size_t> readPins;
    std::set<std::string> writePins;
    std::uint64_t managedBytes = 0;
    std::uint64_t reservedGrowth = 0;
    std::uint64_t reservedTemporaryBytes = 0;
    std::uint64_t freeBytes = 0;
    bool lowSpace = false;
    bool scanInFlight = true;
    bool trimInFlight = false;
    std::size_t retirementsInFlight = 0;
};

PersistentZarrCacheBudget::PersistentZarrCacheBudget(
    fs::path root, Limits limits, SpaceProvider provider)
    : impl_(std::make_unique<Impl>())
{
    impl_->root = std::move(root);
    impl_->limits = limits;
    impl_->spaceProvider = provider ? std::move(provider)
                                    : [](const fs::path& path, std::error_code& ec) {
                                          return fs::space(path, ec);
                                      };
}

std::shared_ptr<PersistentZarrCacheBudget> PersistentZarrCacheBudget::configure(
    const fs::path& root, Limits limits, SpaceProvider provider)
{
    const auto key = normalizedPath(root);
    auto& r = registry();
    std::shared_ptr<PersistentZarrCacheBudget> result;
    bool created = false;
    {
        std::lock_guard lock(r.mutex);
        for (const auto& [registeredRoot, budget] : r.budgets) {
            if (isWithin(key, registeredRoot)) {
                result = budget;
                break;
            }
            if (isWithin(registeredRoot, key)) {
                throw std::invalid_argument(
                    "A persistent Zarr cache budget is already configured beneath " +
                    key.string());
            }
        }
        if (!result) {
            result.reset(new PersistentZarrCacheBudget(key, limits, std::move(provider)));
            r.budgets.emplace(key, result);
            created = true;
        }
    }
    if (created)
        result->startScan();
    else
        result->updateLimits(limits);
    return result;
}

std::shared_ptr<PersistentZarrCacheBudget> PersistentZarrCacheBudget::findForPath(
    const fs::path& path)
{
    const auto normalized = normalizedPath(path);
    auto& r = registry();
    std::lock_guard lock(r.mutex);
    std::shared_ptr<PersistentZarrCacheBudget> best;
    std::size_t bestLength = 0;
    for (const auto& [root, budget] : r.budgets) {
        if (isWithin(normalized, root) && root.native().size() >= bestLength) {
            best = budget;
            bestLength = root.native().size();
        }
    }
    return best;
}

void PersistentZarrCacheBudget::updateAllConfiguredLimits(Limits limits)
{
    std::vector<std::shared_ptr<PersistentZarrCacheBudget>> budgets;
    auto& r = registry();
    {
        std::lock_guard lock(r.mutex);
        for (const auto& [root, budget] : r.budgets) {
            (void)root;
            budgets.push_back(budget);
        }
    }
    for (const auto& budget : budgets)
        budget->updateLimits(limits);
}

const fs::path& PersistentZarrCacheBudget::root() const noexcept
{
    return impl_->root;
}

void PersistentZarrCacheBudget::startScan()
{
    auto self = shared_from_this();
    std::thread([self] {
        std::vector<fs::path> files;
        std::vector<fs::path> artifacts;
        std::vector<NativeZarrArrayLayout> zarrArrays;
        std::vector<fs::path> retiredRoots;
        std::error_code ec;
        fs::recursive_directory_iterator it(
            self->impl_->root, fs::directory_options::skip_permission_denied, ec);
        const fs::recursive_directory_iterator end;
        while (!ec && it != end) {
            if (it->is_directory(ec) && !ec &&
                it->path().filename() == kRetiredCacheDirectory) {
                retiredRoots.push_back(normalizedPath(it->path()));
                it.disable_recursion_pending();
            } else if (it->is_regular_file(ec)) {
                const auto path = normalizedPath(it->path());
                files.push_back(path);
                if (path.filename() == "lasagna-remote.json")
                    artifacts.push_back(path.parent_path());
                if (path.filename() == ".zarray" ||
                    path.filename() == "zarr.json") {
                    if (auto layout = nativeZarrArrayLayout(path))
                        zarrArrays.push_back(std::move(*layout));
                }
            }
            if (ec)
                ec.clear();
            it.increment(ec);
        }

        std::unordered_map<std::string, Impl::Entry> found;
        std::uint64_t total = 0;
        const auto genericPayloads = managedGenericPayloads(files);
        for (const auto& path : files) {
            if (!isVolumeChunk(path, self->impl_->root) &&
                !isLasagnaData(path, artifacts) &&
                !isNativeZarrPayload(path, zarrArrays) &&
                !genericPayloads.contains(path.string()))
                continue;
            std::error_code fileEc;
            const auto size = fs::file_size(path, fileEc);
            if (fileEc)
                continue;
            const auto touched = fs::last_write_time(path, fileEc);
            if (fileEc)
                continue;
            found.emplace(path.string(), Impl::Entry{size, touched});
            total += size;
        }

        {
            std::lock_guard lock(self->impl_->mutex);
            self->impl_->entries = std::move(found);
            self->impl_->managedBytes = total;
            self->impl_->scanInFlight = false;
        }
        for (auto& retired : retiredRoots)
            self->removeRetiredSubtreeAsync(std::move(retired));
        self->startTrim();
        self->impl_->cv.notify_all();
        self->pollSpace();
    }).detach();
}

void PersistentZarrCacheBudget::removeRetiredSubtreeAsync(fs::path retired)
{
    {
        std::lock_guard lock(impl_->mutex);
        ++impl_->retirementsInFlight;
    }
    auto self = shared_from_this();
    std::thread([self, retired = std::move(retired)] {
        std::error_code ec;
        fs::remove_all(retired, ec);
        if (ec) {
            Logger()->warn("Could not delete retired persistent cache {}: {}",
                           retired.string(), ec.message());
        }
        if (retired.filename() != kRetiredCacheDirectory) {
            std::error_code parentEc;
            fs::remove(retired.parent_path(), parentEc);
        }
        {
            std::lock_guard lock(self->impl_->mutex);
            --self->impl_->retirementsInFlight;
        }
        self->impl_->cv.notify_all();
        self->pollSpace();
    }).detach();
}

void PersistentZarrCacheBudget::updateLimits(Limits limits)
{
    bool lower = false;
    {
        std::lock_guard lock(impl_->mutex);
        lower = limits.maximumBytes &&
                (!impl_->limits.maximumBytes ||
                 *limits.maximumBytes < *impl_->limits.maximumBytes);
        impl_->limits = limits;
    }
    pollSpace();
    if (lower)
        startTrim();
}

void PersistentZarrCacheBudget::startTrim()
{
    {
        std::lock_guard lock(impl_->mutex);
        if (impl_->trimInFlight || !impl_->limits.maximumBytes)
            return;
        impl_->trimInFlight = true;
    }
    auto self = shared_from_this();
    std::thread([self] {
        std::unique_lock lock(self->impl_->mutex);
        self->impl_->cv.wait(lock, [&] { return !self->impl_->scanInFlight; });
        while (self->impl_->limits.maximumBytes &&
               self->impl_->managedBytes > *self->impl_->limits.maximumBytes) {
            auto victim = self->impl_->entries.end();
            for (auto it = self->impl_->entries.begin(); it != self->impl_->entries.end(); ++it) {
                if (self->impl_->readPins.contains(it->first) ||
                    self->impl_->writePins.contains(it->first))
                    continue;
                if (victim == self->impl_->entries.end() ||
                    it->second.touched < victim->second.touched)
                    victim = it;
            }
            if (victim == self->impl_->entries.end())
                break;
            const auto path = fs::path(victim->first);
            const auto size = victim->second.size;
            std::error_code ec;
            if (fs::remove(path, ec) && !ec) {
                self->impl_->managedBytes -= std::min(self->impl_->managedBytes, size);
                self->impl_->entries.erase(victim);
            } else {
                Logger()->warn("Could not evict persistent Zarr cache entry {}: {}",
                               path.string(), ec.message());
                // Avoid retrying a failing entry forever in this trim pass.
                victim->second.touched = fs::file_time_type::max();
                bool anyOther = false;
                for (const auto& [key, entry] : self->impl_->entries) {
                    if (key != path.string() && entry.touched != fs::file_time_type::max() &&
                        !self->impl_->readPins.contains(key) &&
                        !self->impl_->writePins.contains(key)) {
                        anyOther = true;
                        break;
                    }
                }
                if (!anyOther)
                    break;
            }
        }
        self->impl_->trimInFlight = false;
        lock.unlock();
        self->impl_->cv.notify_all();
        self->pollSpace();
    }).detach();
}

void PersistentZarrCacheBudget::pollSpace()
{
    std::error_code ec;
    const auto info = impl_->spaceProvider(impl_->root, ec);
    if (ec)
        return;
    std::lock_guard lock(impl_->mutex);
    impl_->freeBytes = info.available;
    impl_->lowSpace = impl_->limits.minimumFreeBytes > 0 &&
                      info.available < impl_->limits.minimumFreeBytes;
}

PersistentZarrCacheBudget::Stats PersistentZarrCacheBudget::stats()
{
    pollSpace();
    std::lock_guard lock(impl_->mutex);
    return Stats{impl_->managedBytes,
                 impl_->freeBytes,
                 impl_->limits.maximumBytes,
                 impl_->limits.minimumFreeBytes,
                 impl_->lowSpace,
                 impl_->scanInFlight,
                 impl_->trimInFlight};
}

PersistentZarrCacheBudget::ReadPin PersistentZarrCacheBudget::pinRead(const fs::path& path)
{
    const auto normalized = normalizedPath(path);
    {
        std::unique_lock lock(impl_->mutex);
        impl_->cv.wait(lock, [&] {
            return !impl_->writePins.contains(normalized.string());
        });
        ++impl_->readPins[normalized.string()];
    }
    return ReadPin(shared_from_this(), normalized);
}

PersistentZarrCacheBudget::WriteReservation PersistentZarrCacheBudget::reserveWrite(
    const fs::path& target, std::uint64_t newSize, std::vector<fs::path> replacements)
{
    return reserveWriteImpl(
        target, newSize, std::move(replacements), true);
}

PersistentZarrCacheBudget::WriteReservation
PersistentZarrCacheBudget::reserveProtectedWrite(
    const fs::path& target, std::uint64_t newSize)
{
    return reserveWriteImpl(target, newSize, {}, false);
}

PersistentZarrCacheBudget::WriteReservation
PersistentZarrCacheBudget::reserveWriteImpl(
    const fs::path& target,
    std::uint64_t newSize,
    std::vector<fs::path> replacements,
    bool managed)
{
    const auto normalizedTarget = normalizedPath(target);
    for (auto& path : replacements)
        path = normalizedPath(path);

    std::unique_lock lock(impl_->mutex);
    impl_->cv.wait(lock, [&] { return !impl_->scanInFlight; });
    impl_->cv.wait(lock, [&] {
        if (impl_->writePins.contains(normalizedTarget.string()) ||
            impl_->readPins.contains(normalizedTarget.string()))
            return false;
        for (const auto& path : replacements) {
            if (impl_->writePins.contains(path.string()) ||
                impl_->readPins.contains(path.string()))
                return false;
        }
        return true;
    });

    const auto trackedSize = [&](const fs::path& path) -> std::uint64_t {
        const auto it = impl_->entries.find(path.string());
        return it == impl_->entries.end() ? 0 : it->second.size;
    };
    std::uint64_t oldSize = 0;
    if (managed) {
        oldSize = trackedSize(normalizedTarget);
        for (const auto& path : replacements)
            oldSize += trackedSize(path);
    }
    const auto netGrowth = managed && newSize > oldSize
        ? newSize - oldSize
        : 0;

    std::error_code spaceEc;
    const auto space = impl_->spaceProvider(impl_->root, spaceEc);
    const std::uint64_t free = spaceEc ? impl_->freeBytes : space.available;
    impl_->freeBytes = free;
    impl_->lowSpace = impl_->limits.minimumFreeBytes > 0 &&
                      free < impl_->limits.minimumFreeBytes;

    std::uint64_t needed = 0;
    if (managed && impl_->limits.maximumBytes) {
        const auto projected = impl_->managedBytes + impl_->reservedGrowth + netGrowth;
        if (projected > *impl_->limits.maximumBytes)
            needed = projected - *impl_->limits.maximumBytes;
    }
    if (impl_->limits.minimumFreeBytes > 0) {
        const auto pendingTemporaryBytes =
            impl_->reservedTemporaryBytes >
                    std::numeric_limits<std::uint64_t>::max() - newSize
                ? std::numeric_limits<std::uint64_t>::max()
                : impl_->reservedTemporaryBytes + newSize;
        if (free < impl_->limits.minimumFreeBytes)
            needed = std::max(needed, pendingTemporaryBytes);
        else if (pendingTemporaryBytes > free - impl_->limits.minimumFreeBytes)
            needed = std::max(needed,
                              pendingTemporaryBytes -
                                  (free - impl_->limits.minimumFreeBytes));
    }

    std::set<std::string> protectedPaths{normalizedTarget.string()};
    for (const auto& path : replacements)
        protectedPaths.insert(path.string());

    if (needed > 0) {
        struct Candidate {
            std::string path;
            std::uint64_t size;
            fs::file_time_type touched;
        };
        std::vector<Candidate> candidates;
        std::uint64_t evictable = 0;
        for (const auto& [path, entry] : impl_->entries) {
            if (protectedPaths.contains(path) || impl_->readPins.contains(path) ||
                impl_->writePins.contains(path))
                continue;
            candidates.push_back({path, entry.size, entry.touched});
            evictable = entry.size >
                                std::numeric_limits<std::uint64_t>::max() - evictable
                            ? std::numeric_limits<std::uint64_t>::max()
                            : evictable + entry.size;
        }
        if (evictable < needed)
            return {};
        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate& a, const Candidate& b) {
                      return a.touched < b.touched;
                  });

        std::uint64_t evicted = 0;
        for (const auto& candidate : candidates) {
            if (evicted >= needed)
                break;
            const auto victimPath = fs::path(candidate.path);
            std::error_code ec;
            if (fs::remove(victimPath, ec) && !ec) {
                evicted += candidate.size;
                impl_->managedBytes -= std::min(impl_->managedBytes, candidate.size);
                impl_->entries.erase(candidate.path);
            } else {
                Logger()->warn("Could not evict persistent Zarr cache entry {}: {}",
                               victimPath.string(), ec.message());
            }
        }
        if (evicted < needed)
            return {};
    }

    impl_->reservedGrowth += netGrowth;
    impl_->reservedTemporaryBytes += newSize;
    impl_->writePins.insert(normalizedTarget.string());
    for (const auto& path : replacements)
        impl_->writePins.insert(path.string());
    return WriteReservation(shared_from_this(), normalizedTarget,
                            std::move(replacements), netGrowth, newSize,
                            managed);
}

bool PersistentZarrCacheBudget::removeCacheSubtree(
    const fs::path& subtree, std::error_code& ec)
{
    const auto normalized = normalizedPath(subtree);
    if (normalized == impl_->root || !isWithin(normalized, impl_->root)) {
        ec = std::make_error_code(std::errc::invalid_argument);
        return false;
    }

    const auto overlaps = [&](const std::string& path) {
        return isWithin(fs::path(path), normalized);
    };
    std::unique_lock lock(impl_->mutex);
    impl_->cv.wait(lock, [&] {
        if (impl_->scanInFlight || impl_->trimInFlight)
            return false;
        for (const auto& [path, count] : impl_->readPins) {
            (void)count;
            if (overlaps(path))
                return false;
        }
        return std::none_of(
            impl_->writePins.begin(), impl_->writePins.end(), overlaps);
    });

    std::optional<fs::path> retired;
    if (fs::exists(normalized, ec)) {
        const auto retiredRoot = normalized.parent_path() /
                                 kRetiredCacheDirectory;
        fs::create_directories(retiredRoot, ec);
        if (ec)
            return false;
        retired = retiredCachePath(normalized);
        fs::rename(normalized, *retired, ec);
        if (ec)
            return false;
    } else if (ec) {
        return false;
    }

    for (auto it = impl_->entries.begin(); it != impl_->entries.end();) {
        if (!overlaps(it->first)) {
            ++it;
            continue;
        }
        impl_->managedBytes -=
            std::min(impl_->managedBytes, it->second.size);
        it = impl_->entries.erase(it);
    }
    lock.unlock();
    if (retired)
        removeRetiredSubtreeAsync(std::move(*retired));
    else
        pollSpace();
    return true;
}

bool PersistentZarrCacheBudget::moveCacheSubtree(
    const fs::path& source,
    const fs::path& destination,
    std::error_code& ec)
{
    const auto normalizedSource = normalizedPath(source);
    const auto normalizedDestination = normalizedPath(destination);
    ec.clear();
    if (normalizedSource == normalizedDestination)
        return true;
    if (normalizedSource == impl_->root ||
        normalizedDestination == impl_->root ||
        !isWithin(normalizedSource, impl_->root) ||
        !isWithin(normalizedDestination, impl_->root) ||
        isWithin(normalizedSource, normalizedDestination) ||
        isWithin(normalizedDestination, normalizedSource)) {
        ec = std::make_error_code(std::errc::invalid_argument);
        return false;
    }

    const auto overlapsEither = [&](const std::string& path) {
        const auto candidate = fs::path(path);
        return isWithin(candidate, normalizedSource) ||
               isWithin(candidate, normalizedDestination);
    };
    std::unique_lock lock(impl_->mutex);
    impl_->cv.wait(lock, [&] {
        if (impl_->scanInFlight || impl_->trimInFlight)
            return false;
        for (const auto& [path, count] : impl_->readPins) {
            (void)count;
            if (overlapsEither(path))
                return false;
        }
        return std::none_of(
            impl_->writePins.begin(), impl_->writePins.end(), overlapsEither);
    });

    if (!fs::exists(normalizedSource, ec))
        return !ec;
    if (fs::exists(normalizedDestination, ec)) {
        ec = std::make_error_code(std::errc::file_exists);
        return false;
    }
    if (ec)
        return false;
    fs::create_directories(normalizedDestination.parent_path(), ec);
    if (ec)
        return false;
    fs::rename(normalizedSource, normalizedDestination, ec);
    if (ec)
        return false;

    std::vector<std::pair<std::string, Impl::Entry>> movedEntries;
    for (auto it = impl_->entries.begin(); it != impl_->entries.end();) {
        const auto path = fs::path(it->first);
        if (!isWithin(path, normalizedSource)) {
            ++it;
            continue;
        }
        movedEntries.emplace_back(
            (normalizedDestination /
             path.lexically_relative(normalizedSource)).string(),
            it->second);
        it = impl_->entries.erase(it);
    }
    for (auto& [path, entry] : movedEntries)
        impl_->entries.insert_or_assign(std::move(path), std::move(entry));
    lock.unlock();
    pollSpace();
    return true;
}

void PersistentZarrCacheBudget::releaseRead(const fs::path& path, bool touch)
{
    if (touch) {
        std::error_code ec;
        const auto now = fs::file_time_type::clock::now();
        fs::last_write_time(path, now, ec);
        if (!ec) {
            std::lock_guard lock(impl_->mutex);
            if (auto it = impl_->entries.find(path.string()); it != impl_->entries.end())
                it->second.touched = now;
        }
    }
    bool needsTrim = false;
    {
        std::lock_guard lock(impl_->mutex);
        if (auto it = impl_->readPins.find(path.string()); it != impl_->readPins.end()) {
            if (--it->second == 0)
                impl_->readPins.erase(it);
        }
        needsTrim = impl_->limits.maximumBytes &&
                    impl_->managedBytes > *impl_->limits.maximumBytes;
    }
    impl_->cv.notify_all();
    if (needsTrim)
        startTrim();
}

void PersistentZarrCacheBudget::finishWrite(
    const fs::path& target, const std::vector<fs::path>& replacements,
    std::uint64_t reservedGrowth, std::uint64_t reservedTemporaryBytes,
    bool committed, bool managed)
{
    bool needsTrim = false;
    {
        std::lock_guard lock(impl_->mutex);
        impl_->reservedGrowth -= std::min(impl_->reservedGrowth, reservedGrowth);
        impl_->reservedTemporaryBytes -=
            std::min(impl_->reservedTemporaryBytes, reservedTemporaryBytes);
        impl_->writePins.erase(target.string());
        for (const auto& path : replacements)
            impl_->writePins.erase(path.string());
        if (committed) {
            auto refresh = [&](const fs::path& path, bool track) {
                const auto key = path.string();
                if (auto it = impl_->entries.find(key); it != impl_->entries.end()) {
                    impl_->managedBytes -= std::min(impl_->managedBytes, it->second.size);
                    impl_->entries.erase(it);
                }
                if (!track)
                    return;
                std::error_code ec;
                if (!fs::is_regular_file(path, ec) || ec)
                    return;
                const auto size = fs::file_size(path, ec);
                if (ec)
                    return;
                const auto touched = fs::last_write_time(path, ec);
                if (ec)
                    return;
                impl_->entries.emplace(key, Impl::Entry{size, touched});
                impl_->managedBytes += size;
            };
            refresh(target, managed);
            for (const auto& path : replacements)
                refresh(path, managed);
        }
        needsTrim = impl_->limits.maximumBytes &&
                    impl_->managedBytes > *impl_->limits.maximumBytes;
    }
    impl_->cv.notify_all();
    if (needsTrim)
        startTrim();
}

void PersistentZarrCacheBudget::waitForIdle()
{
    std::unique_lock lock(impl_->mutex);
    impl_->cv.wait(lock, [&] {
        return !impl_->scanInFlight && !impl_->trimInFlight &&
               impl_->writePins.empty() && impl_->retirementsInFlight == 0;
    });
}

void PersistentZarrCacheBudget::resetRegistryForTesting()
{
    auto& r = registry();
    std::lock_guard lock(r.mutex);
    r.budgets.clear();
}

PersistentZarrCacheBudget::ReadPin::ReadPin(
    std::shared_ptr<PersistentZarrCacheBudget> owner, fs::path path)
    : owner_(std::move(owner)), path_(std::move(path)) {}

PersistentZarrCacheBudget::ReadPin::ReadPin(ReadPin&& other) noexcept = default;
PersistentZarrCacheBudget::ReadPin& PersistentZarrCacheBudget::ReadPin::operator=(ReadPin&& other) noexcept
{
    if (this != &other) {
        release(false);
        owner_ = std::move(other.owner_);
        path_ = std::move(other.path_);
    }
    return *this;
}
PersistentZarrCacheBudget::ReadPin::~ReadPin() { release(false); }
void PersistentZarrCacheBudget::ReadPin::complete(bool successful) { release(successful); }
void PersistentZarrCacheBudget::ReadPin::release(bool touch)
{
    if (owner_) {
        auto owner = std::move(owner_);
        owner->releaseRead(path_, touch);
    }
}

PersistentZarrCacheBudget::WriteReservation::WriteReservation(
    std::shared_ptr<PersistentZarrCacheBudget> owner, fs::path target,
    std::vector<fs::path> replacements, std::uint64_t growth,
    std::uint64_t temporaryBytes, bool managed)
    : owner_(std::move(owner)), target_(std::move(target)),
      replacements_(std::move(replacements)), reservedGrowth_(growth),
      reservedTemporaryBytes_(temporaryBytes), managed_(managed) {}
PersistentZarrCacheBudget::WriteReservation::WriteReservation(WriteReservation&& other) noexcept = default;
PersistentZarrCacheBudget::WriteReservation& PersistentZarrCacheBudget::WriteReservation::operator=(WriteReservation&& other) noexcept
{
    if (this != &other) {
        cancel();
        owner_ = std::move(other.owner_);
        target_ = std::move(other.target_);
        replacements_ = std::move(other.replacements_);
        reservedGrowth_ = other.reservedGrowth_;
        reservedTemporaryBytes_ = other.reservedTemporaryBytes_;
        managed_ = other.managed_;
        other.reservedGrowth_ = 0;
        other.reservedTemporaryBytes_ = 0;
    }
    return *this;
}
PersistentZarrCacheBudget::WriteReservation::~WriteReservation() { cancel(); }
void PersistentZarrCacheBudget::WriteReservation::commit()
{
    if (owner_) {
        auto owner = std::move(owner_);
        owner->finishWrite(target_, replacements_, reservedGrowth_,
                           reservedTemporaryBytes_, true, managed_);
    }
}
void PersistentZarrCacheBudget::WriteReservation::cancel()
{
    if (owner_) {
        auto owner = std::move(owner_);
        owner->finishWrite(target_, replacements_, reservedGrowth_,
                           reservedTemporaryBytes_, false, managed_);
    }
}

} // namespace vc::render
