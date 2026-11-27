// Shared budget registry for volume and hosted-object caches.
#include "disk_cache.hpp"

namespace vcr::zarr {
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
    // One budget per directory, whichever spelling names it: readers pass resolved
    // roots, other callers may pass a symlinked one (macOS /var -> /private/var).
    const auto key = utf8String(std::filesystem::weakly_canonical(utf8Path(root)));
    auto& budget = budgets[key];
    if (!budget)
        budget = std::make_shared<DiskCache>(key, maxBytes.value_or(-1));
    else if (maxBytes)
        budget->set_max_bytes(*maxBytes);
    return budget;
}

} // namespace vcr::zarr
