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
    // One budget per directory, however it is spelled: a symlinked parent (macOS's /tmp
    // and /var), a trailing slash or a relative path would otherwise each get their own,
    // and a limit or free-space floor set through one would not apply to the others.
    std::error_code ec;
    auto key = std::filesystem::weakly_canonical(root, ec).string();
    if (ec || key.empty())
        key = root;
    auto& budget = budgets[key];
    if (!budget)
        budget = std::make_shared<DiskCache>(root, maxBytes.value_or(-1));
    else if (maxBytes)
        budget->set_max_bytes(*maxBytes);
    return budget;
}

} // namespace vcr::zarr
