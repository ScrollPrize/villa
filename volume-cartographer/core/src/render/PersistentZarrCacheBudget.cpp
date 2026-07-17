#include "vc/core/render/PersistentZarrCacheBudget.hpp"

#include "vc/core/util/Logging.hpp"

#include <algorithm>
#include <condition_variable>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>

namespace vc::render {
namespace fs = std::filesystem;
namespace {

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
    if (ext != ".bin" && ext != ".zst" && ext != ".c3d" && ext != ".empty")
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
    std::uint64_t freeBytes = 0;
    bool lowSpace = false;
    bool scanInFlight = true;
    bool trimInFlight = false;
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
        std::error_code ec;
        fs::recursive_directory_iterator it(
            self->impl_->root, fs::directory_options::skip_permission_denied, ec);
        const fs::recursive_directory_iterator end;
        while (!ec && it != end) {
            if (it->is_regular_file(ec)) {
                const auto path = normalizedPath(it->path());
                files.push_back(path);
                if (path.filename() == "lasagna-remote.json")
                    artifacts.push_back(path.parent_path());
            }
            if (ec)
                ec.clear();
            it.increment(ec);
        }

        std::unordered_map<std::string, Impl::Entry> found;
        std::uint64_t total = 0;
        for (const auto& path : files) {
            if (!isVolumeChunk(path, self->impl_->root) &&
                !isLasagnaData(path, artifacts))
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
        self->startTrim();
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
    std::uint64_t oldSize = trackedSize(normalizedTarget);
    for (const auto& path : replacements)
        oldSize += trackedSize(path);
    const auto netGrowth = newSize > oldSize ? newSize - oldSize : 0;

    std::error_code spaceEc;
    const auto space = impl_->spaceProvider(impl_->root, spaceEc);
    const std::uint64_t free = spaceEc ? impl_->freeBytes : space.available;
    impl_->freeBytes = free;
    impl_->lowSpace = impl_->limits.minimumFreeBytes > 0 &&
                      free < impl_->limits.minimumFreeBytes;

    std::uint64_t needed = 0;
    if (impl_->limits.maximumBytes) {
        const auto projected = impl_->managedBytes + impl_->reservedGrowth + netGrowth;
        if (projected > *impl_->limits.maximumBytes)
            needed = projected - *impl_->limits.maximumBytes;
    }
    if (impl_->limits.minimumFreeBytes > 0) {
        const auto pendingGrowth =
            impl_->reservedGrowth > std::numeric_limits<std::uint64_t>::max() - netGrowth
                ? std::numeric_limits<std::uint64_t>::max()
                : impl_->reservedGrowth + netGrowth;
        if (free < impl_->limits.minimumFreeBytes)
            needed = std::max(needed, pendingGrowth);
        else if (pendingGrowth > free - impl_->limits.minimumFreeBytes)
            needed = std::max(needed,
                              pendingGrowth - (free - impl_->limits.minimumFreeBytes));
    }

    std::set<std::string> protectedPaths{normalizedTarget.string()};
    for (const auto& path : replacements)
        protectedPaths.insert(path.string());
    std::uint64_t evicted = 0;
    while (evicted < needed) {
        auto victim = impl_->entries.end();
        for (auto it = impl_->entries.begin(); it != impl_->entries.end(); ++it) {
            if (protectedPaths.contains(it->first) || impl_->readPins.contains(it->first) ||
                impl_->writePins.contains(it->first))
                continue;
            if (victim == impl_->entries.end() || it->second.touched < victim->second.touched)
                victim = it;
        }
        if (victim == impl_->entries.end())
            break;
        const auto victimPath = fs::path(victim->first);
        const auto size = victim->second.size;
        std::error_code ec;
        if (fs::remove(victimPath, ec) && !ec) {
            evicted += size;
            impl_->managedBytes -= std::min(impl_->managedBytes, size);
            impl_->entries.erase(victim);
        } else {
            Logger()->warn("Could not evict persistent Zarr cache entry {}: {}",
                           victimPath.string(), ec.message());
            // Pin for this admission attempt and try another candidate.
            protectedPaths.insert(victimPath.string());
        }
    }
    if (evicted < needed)
        return {};

    impl_->reservedGrowth += netGrowth;
    impl_->writePins.insert(normalizedTarget.string());
    for (const auto& path : replacements)
        impl_->writePins.insert(path.string());
    return WriteReservation(shared_from_this(), normalizedTarget,
                            std::move(replacements), netGrowth);
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
    std::uint64_t reservedGrowth, bool committed)
{
    bool needsTrim = false;
    {
        std::lock_guard lock(impl_->mutex);
        impl_->reservedGrowth -= std::min(impl_->reservedGrowth, reservedGrowth);
        impl_->writePins.erase(target.string());
        for (const auto& path : replacements)
            impl_->writePins.erase(path.string());
        if (committed) {
            auto refresh = [&](const fs::path& path) {
                const auto key = path.string();
                if (auto it = impl_->entries.find(key); it != impl_->entries.end()) {
                    impl_->managedBytes -= std::min(impl_->managedBytes, it->second.size);
                    impl_->entries.erase(it);
                }
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
            refresh(target);
            for (const auto& path : replacements)
                refresh(path);
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
               impl_->writePins.empty();
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
    std::vector<fs::path> replacements, std::uint64_t growth)
    : owner_(std::move(owner)), target_(std::move(target)),
      replacements_(std::move(replacements)), reservedGrowth_(growth) {}
PersistentZarrCacheBudget::WriteReservation::WriteReservation(WriteReservation&& other) noexcept = default;
PersistentZarrCacheBudget::WriteReservation& PersistentZarrCacheBudget::WriteReservation::operator=(WriteReservation&& other) noexcept
{
    if (this != &other) {
        cancel();
        owner_ = std::move(other.owner_);
        target_ = std::move(other.target_);
        replacements_ = std::move(other.replacements_);
        reservedGrowth_ = other.reservedGrowth_;
        other.reservedGrowth_ = 0;
    }
    return *this;
}
PersistentZarrCacheBudget::WriteReservation::~WriteReservation() { cancel(); }
void PersistentZarrCacheBudget::WriteReservation::commit()
{
    if (owner_) {
        auto owner = std::move(owner_);
        owner->finishWrite(target_, replacements_, reservedGrowth_, true);
    }
}
void PersistentZarrCacheBudget::WriteReservation::cancel()
{
    if (owner_) {
        auto owner = std::move(owner_);
        owner->finishWrite(target_, replacements_, reservedGrowth_, false);
    }
}

} // namespace vc::render
