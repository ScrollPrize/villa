#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>

namespace vc::objects::util {
struct ProgressUpdate {
    std::string_view stage;
    std::uint64_t completed = 0;
    std::uint64_t total = 0; // zero means unknown, not a fabricated percentage
    std::string_view detail;
};
// Views are valid for the duration of the call. Callbacks run synchronously on
// the reporting thread; consumers supporting parallel producers must synchronize.
using ProgressCallback = std::function<void(const ProgressUpdate&)>;
inline void reportProgress(const ProgressCallback& callback, std::string_view stage,
    std::uint64_t completed = 0, std::uint64_t total = 0, std::string_view detail = {})
{
    if (callback) callback({stage, completed, total, detail});
}

// Keep one operation's counters stable while nested work reports its activity.
// Own the labels: the returned callback may outlive the strings passed here.
inline ProgressCallback subtaskProgress(const ProgressCallback& callback,
    std::string stage, std::uint64_t completed, std::uint64_t total, std::string item)
{
    if (!callback) return {};
    return [=](const ProgressUpdate& event) {
        std::string detail = item + ": " + std::string(event.stage);
        if (event.total) {
            const auto percent = event.completed >= event.total ? 100 :
                static_cast<unsigned>(100.0 * double(event.completed) / double(event.total));
            detail += " " + std::to_string(percent) + "%";
        }
        if (!event.detail.empty()) detail += " " + std::string(event.detail);
        reportProgress(callback, stage, completed, total, detail);
    };
}
} // namespace vc::objects::util
