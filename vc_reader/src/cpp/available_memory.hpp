// Available RAM for an adaptive decoded cache, without counting swap.
#pragma once
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef __APPLE__
#include <mach/mach.h>
#endif

namespace vcr::memory {
inline std::optional<std::uint64_t> number(const std::filesystem::path& path)
{
    std::ifstream in(path);
    std::uint64_t value;
    if (in >> value) return value;
    return std::nullopt; // includes the cgroup v2 "max" sentinel
}

inline std::optional<std::uint64_t> field(const std::filesystem::path& path, const std::string& key)
{
    std::ifstream in(path);
    std::string line, name;
    std::uint64_t value;
    while (std::getline(in, line)) {
        std::istringstream row(line);
        if (row >> name >> value && name == key) return value;
    }
    return std::nullopt;
}

struct Cgroup { std::filesystem::path directory; bool v2; };

inline std::string mount_path(std::string value)
{
    // mountinfo quotes whitespace and backslashes as octal escapes.
    for (std::size_t p = 0; p + 3 < value.size(); ++p) {
        if (value[p] == '\\' && value[p+1] >= '0' && value[p+1] <= '7' &&
            value[p+2] >= '0' && value[p+2] <= '7' && value[p+3] >= '0' && value[p+3] <= '7') {
            const char decoded = char((value[p+1]-'0')*64 + (value[p+2]-'0')*8 + value[p+3]-'0');
            value.replace(p, 4, 1, decoded);
        }
    }
    return value;
}

inline std::vector<Cgroup> cgroups()
{
    std::string unified, memory;
    std::ifstream groups("/proc/self/cgroup");
    std::string line;
    while (std::getline(groups, line)) {
        const auto a = line.find(':'), b = line.find(':', a == std::string::npos ? a : a+1);
        if (b == std::string::npos) continue;
        const auto controllers = "," + line.substr(a+1, b-a-1) + ",";
        if (a+1 == b) unified = line.substr(b+1);
        if (controllers.find(",memory,") != std::string::npos) memory = line.substr(b+1);
    }
    std::vector<Cgroup> result;
    std::ifstream mounts("/proc/self/mountinfo");
    while (std::getline(mounts, line)) {
        const auto split = line.find(" - ");
        if (split == std::string::npos) continue;
        std::istringstream after(line.substr(split+3));
        std::string type, source, options;
        after >> type >> source >> options;
        const bool v2 = type == "cgroup2";
        if (!v2 && (type != "cgroup" || (","+options+",").find(",memory,") == std::string::npos)) continue;
        const auto& membership = v2 ? unified : memory;
        if (membership.empty()) continue;
        std::istringstream before(line.substr(0, split));
        std::string id, parent, device, root, target;
        before >> id >> parent >> device >> root >> target;
        root = mount_path(root); target = mount_path(target);
        auto relative = std::filesystem::path(membership).lexically_relative(root);
        // A cgroup namespace commonly reports '/' even for a mounted subtree.
        if (membership == "/") relative.clear();
        if (relative.empty() && membership != root && membership != "/") continue;
        if (!relative.empty() && *relative.begin() == "..") continue;
        const auto mount = std::filesystem::path(target).lexically_normal();
        auto current = (mount / relative).lexically_normal();
        for (;;) {
            result.push_back({current, v2});
            if (current == mount || current == current.parent_path()) break;
            current = current.parent_path();
        }
    }
    return result;
}

inline std::uint64_t cgroup_available(std::uint64_t limit, std::uint64_t usage, std::uint64_t inactive_file)
{
    const auto used = usage - std::min(usage, inactive_file);
    return limit > used ? limit-used : 0;
}

inline std::uint64_t available_bytes()
{
#ifdef __linux__
    const auto kb = field("/proc/meminfo", "MemAvailable:");
    if (!kb) throw std::runtime_error("auto cache requires /proc/meminfo MemAvailable");
    auto available = *kb * 1024;
    // Check every ancestor's limit; limiting only the leaf misses containers
    // under a constrained parent. Refresh membership too (fork/migration).
    for (const auto& group : cgroups()) {
        const auto limit = number(group.directory / (group.v2 ? "memory.max" : "memory.limit_in_bytes"));
        const auto usage = number(group.directory / (group.v2 ? "memory.current" : "memory.usage_in_bytes"));
        if (!limit || !usage) continue;
        const auto reclaimable = field(group.directory / "memory.stat", group.v2 ? "inactive_file" : "total_inactive_file").value_or(0);
        available = std::min(available, cgroup_available(*limit, *usage, reclaimable));
    }
    return available;
#elif defined(__APPLE__)
    const auto host = mach_host_self();
    vm_size_t page = 0;
    vm_statistics64_data_t stats{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    const auto page_result = host_page_size(host, &page);
    const auto result = host_statistics64(host, HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&stats), &count);
    mach_port_deallocate(mach_task_self(), host);
    if (page_result != KERN_SUCCESS || result != KERN_SUCCESS)
        throw std::runtime_error("cannot query available memory for auto cache");
    // Conservative reclaimable estimate; do not count compressed/swap memory.
    return (std::uint64_t(stats.free_count) + stats.inactive_count) * page;
#else
    throw std::runtime_error("auto memory cache is supported on Linux and macOS");
#endif
}
} // namespace vcr::memory
