#pragma once
#include <cstdint>
#include <functional>
#include <string_view>

namespace vc::util {
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
} // namespace vc::util
