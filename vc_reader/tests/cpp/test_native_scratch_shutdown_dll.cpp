#include <vc_reader/native/scratch_pool.hpp>
#include <chrono>
#include <thread>

extern "C" __declspec(dllexport) bool exerciseScratch()
{
    auto stats = vcr::scratchMaintenance().stats;
    {
        auto lease = vcr::workerScratch().lease();
        auto bytes = lease.bytes(0, 4096);
        bytes.front() = 17;
        bytes.back() = 23;
        if (bytes.front() + bytes.back() != 40)
            return false;
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (stats->bytes.load() != 0 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    return stats->bytes.load() == 0;
}
