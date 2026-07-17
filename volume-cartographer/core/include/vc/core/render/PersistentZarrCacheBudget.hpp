#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <system_error>
#include <vector>

namespace vc::render {

// Process-wide accounting and eviction for persistent remote Zarr chunks.
// Instances are registered per configured cache root; callers outside a
// registered root keep their historical unlimited behaviour.
class PersistentZarrCacheBudget final
    : public std::enable_shared_from_this<PersistentZarrCacheBudget> {
public:
    struct Limits {
        std::optional<std::uint64_t> maximumBytes;
        std::uint64_t minimumFreeBytes = 0;
    };

    struct Stats {
        std::uint64_t managedBytes = 0;
        std::uint64_t freeBytes = 0;
        std::optional<std::uint64_t> maximumBytes;
        std::uint64_t minimumFreeBytes = 0;
        bool lowSpace = false;
        bool scanInFlight = false;
        bool trimInFlight = false;
    };

    using SpaceProvider = std::function<std::filesystem::space_info(
        const std::filesystem::path&, std::error_code&)>;

    class ReadPin {
    public:
        ReadPin() = default;
        ReadPin(ReadPin&& other) noexcept;
        ReadPin& operator=(ReadPin&& other) noexcept;
        ReadPin(const ReadPin&) = delete;
        ReadPin& operator=(const ReadPin&) = delete;
        ~ReadPin();

        // Successful reads update the persistent approximate-LRU timestamp.
        void complete(bool successful = true);

    private:
        friend class PersistentZarrCacheBudget;
        ReadPin(std::shared_ptr<PersistentZarrCacheBudget> owner,
                std::filesystem::path path);
        void release(bool touch);
        std::shared_ptr<PersistentZarrCacheBudget> owner_;
        std::filesystem::path path_;
    };

    class WriteReservation {
    public:
        WriteReservation() = default;
        WriteReservation(WriteReservation&& other) noexcept;
        WriteReservation& operator=(WriteReservation&& other) noexcept;
        WriteReservation(const WriteReservation&) = delete;
        WriteReservation& operator=(const WriteReservation&) = delete;
        ~WriteReservation();

        explicit operator bool() const noexcept { return owner_ != nullptr; }
        // Call only after the target has been atomically published and any
        // replacement paths have been removed.
        void commit();
        void cancel();

    private:
        friend class PersistentZarrCacheBudget;
        WriteReservation(std::shared_ptr<PersistentZarrCacheBudget> owner,
                         std::filesystem::path target,
                         std::vector<std::filesystem::path> replacements,
                         std::uint64_t reservedGrowth);
        std::shared_ptr<PersistentZarrCacheBudget> owner_;
        std::filesystem::path target_;
        std::vector<std::filesystem::path> replacements_;
        std::uint64_t reservedGrowth_ = 0;
    };

    static std::shared_ptr<PersistentZarrCacheBudget> configure(
        const std::filesystem::path& root,
        Limits limits,
        SpaceProvider spaceProvider = {});
    static std::shared_ptr<PersistentZarrCacheBudget> findForPath(
        const std::filesystem::path& path);
    static void updateAllConfiguredLimits(Limits limits);

    const std::filesystem::path& root() const noexcept;
    void updateLimits(Limits limits);
    Stats stats();
    void pollSpace();

    ReadPin pinRead(const std::filesystem::path& path);
    WriteReservation reserveWrite(
        const std::filesystem::path& target,
        std::uint64_t newSize,
        std::vector<std::filesystem::path> replacedPaths = {});

    // Primarily useful for deterministic tests and orderly shutdown checks.
    void waitForIdle();
    static void resetRegistryForTesting();

private:
    struct Impl;
    PersistentZarrCacheBudget(std::filesystem::path root,
                              Limits limits,
                              SpaceProvider spaceProvider);
    void startScan();
    void startTrim();
    void releaseRead(const std::filesystem::path& path, bool touch);
    void finishWrite(const std::filesystem::path& target,
                     const std::vector<std::filesystem::path>& replacements,
                     std::uint64_t reservedGrowth,
                     bool committed);

    std::unique_ptr<Impl> impl_;
};

} // namespace vc::render
