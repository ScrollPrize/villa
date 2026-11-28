// Shared budget registry for volume and hosted-object caches.
#include "disk_cache.hpp"
#include "zarr/files.hpp"

namespace vcr::zarr {
namespace {
std::shared_ptr<DiskCache> accessDiskBudget(const std::string& root, std::optional<std::int64_t> maxBytes, bool release)
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
    if (release) {
        budgets.erase(root);
        return {};
    }
    auto& budget = budgets[root];
    if (!budget)
        budget = std::make_shared<DiskCache>(root, maxBytes.value_or(-1));
    else if (maxBytes)
        budget->set_max_bytes(*maxBytes);
    return budget;
}
}

std::shared_ptr<DiskCache> diskBudget(const std::string& root, std::optional<std::int64_t> maxBytes)
{
    return accessDiskBudget(root, maxBytes, false);
}

void detail::releaseDiskBudget(const std::string& root)
{
    accessDiskBudget(root, std::nullopt, true);
}

} // namespace vcr::zarr
