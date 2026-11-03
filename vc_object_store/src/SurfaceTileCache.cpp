#include "vc_object_store/objects/SurfaceTileCache.hpp"
#include "geometry/GeometryReader.hpp"
#include <list>
#include <mutex>
#include <tuple>
#include <unordered_map>

namespace vc::objects {
namespace {
struct Key {
    std::string revision;
    int64_t row, column, rows, columns;
    bool operator==(const Key&) const = default;
};
struct KeyHash {
    size_t operator()(const Key& k) const {
        size_t h = std::hash<std::string>{}(k.revision);
        for (const auto v : {k.row, k.column, k.rows, k.columns})
            h ^= std::hash<int64_t>{}(v) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        return h;
    }
};
size_t charge(const Key& key, const detail::SurfaceTile& tile)
{
    // Payload plus the entry's bookkeeping (list node, map node, key).
    return tile.points.size() * sizeof(tile.points[0]) + tile.valid.size() +
           tile.bands.size() * sizeof(tile.bands[0]) + key.revision.size() + 192;
}
}

struct SurfaceTileCache::Impl {
    using Entry = std::pair<Key, std::shared_ptr<const detail::SurfaceTile>>;
    mutable std::mutex mutex;
    // Most recently used first.
    mutable std::list<Entry> order;
    std::unordered_map<Key, std::list<Entry>::iterator, KeyHash> entries;
    size_t capacity = 0, bytes = 0;

    void evict() {
        while (bytes > capacity && !order.empty()) {
            const auto& [key, tile] = order.back();
            bytes -= charge(key, *tile);
            entries.erase(key);
            order.pop_back();
        }
    }
};

SurfaceTileCache::SurfaceTileCache(size_t capacityBytes) : impl_(std::make_unique<Impl>())
{
    impl_->capacity = capacityBytes;
}

SurfaceTileCache::~SurfaceTileCache() = default;

void SurfaceTileCache::setCapacity(size_t bytes)
{
    std::lock_guard lock(impl_->mutex);
    impl_->capacity = bytes;
    impl_->evict();
}

size_t SurfaceTileCache::capacity() const
{
    std::lock_guard lock(impl_->mutex);
    return impl_->capacity;
}

size_t SurfaceTileCache::size() const
{
    std::lock_guard lock(impl_->mutex);
    return impl_->bytes;
}

std::shared_ptr<const detail::SurfaceTile> SurfaceTileCache::find(const std::string& revision, int64_t row,
                                                                  int64_t column, int64_t rows, int64_t columns) const
{
    std::lock_guard lock(impl_->mutex);
    const auto found = impl_->entries.find(Key{revision, row, column, rows, columns});
    if (found == impl_->entries.end()) return nullptr;
    impl_->order.splice(impl_->order.begin(), impl_->order, found->second);
    return found->second->second;
}

void SurfaceTileCache::insert(const std::string& revision, int64_t row, int64_t column, int64_t rows,
                              int64_t columns, std::shared_ptr<const detail::SurfaceTile> tile)
{
    if (!tile) return;
    Key key{revision, row, column, rows, columns};
    const auto bytes = charge(key, *tile);
    std::lock_guard lock(impl_->mutex);
    if (bytes > impl_->capacity || impl_->entries.contains(key)) return;
    impl_->order.emplace_front(key, std::move(tile));
    impl_->entries.emplace(std::move(key), impl_->order.begin());
    impl_->bytes += bytes;
    impl_->evict();
}
}
