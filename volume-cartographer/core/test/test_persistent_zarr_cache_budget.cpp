#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/render/PersistentZarrCacheBudget.hpp"

#include <atomic>
#include <chrono>
#include <fstream>
#include <random>
#include <thread>

namespace fs = std::filesystem;
using Budget = vc::render::PersistentZarrCacheBudget;

namespace {

fs::path tempRoot(const char* tag)
{
    static std::atomic<unsigned long long> serial{0};
    auto root = fs::temp_directory_path() /
        (std::string("vc_zarr_budget_") + tag + "_" +
         std::to_string(serial.fetch_add(1)));
    fs::create_directories(root);
    return root;
}

void writeBytes(const fs::path& path, std::size_t size, char value = 'x')
{
    fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    std::string bytes(size, value);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

fs::path volumeChunk(const fs::path& root, int x)
{
    return root / "volume-hash" / "level_0" / "0" / "0" /
           (std::to_string(x) + ".bin");
}

Budget::SpaceProvider spaceWith(std::shared_ptr<std::atomic<std::uint64_t>> free)
{
    return [free](const fs::path&, std::error_code& ec) {
        ec.clear();
        return fs::space_info{1000, free->load(), free->load()};
    };
}

bool publish(Budget& budget, const fs::path& path, std::size_t size)
{
    auto reservation = budget.reserveWrite(path, size);
    if (!reservation)
        return false;
    writeBytes(path, size);
    reservation.commit();
    return true;
}

} // namespace

TEST_CASE("budget discovers only remote-volume and Lasagna Zarr data")
{
    const auto root = tempRoot("discover");
    writeBytes(volumeChunk(root, 0), 11);
    writeBytes(root / "volume-hash" / "manifest.json", 19);
    writeBytes(root / "segments" / "level_0" / "0" / "0" / "0.bin", 23);
    writeBytes(root / "normal_grids" / "level_0" / "0" / "0" / "0.zst", 29);

    const auto lasagna = root / "open_data" / "lasagna" / "artifact";
    writeBytes(lasagna / "lasagna-remote.json", 2);
    writeBytes(lasagna / "model.lasagna.json", 31);
    writeBytes(lasagna / ".lasagna-zarr-metadata" / "group" / ".zarray", 37);
    writeBytes(lasagna / "group" / "0.0.0", 13);

    auto free = std::make_shared<std::atomic<std::uint64_t>>(900);
    auto budget = Budget::configure(root, {}, spaceWith(free));
    budget->waitForIdle();
    CHECK(budget->stats().managedBytes == 24);
    fs::remove_all(root);
}

TEST_CASE("successful reads touch LRU and the oldest unpinned entry is evicted")
{
    const auto root = tempRoot("lru");
    const auto old = volumeChunk(root, 0);
    const auto recent = volumeChunk(root, 1);
    writeBytes(old, 10);
    writeBytes(recent, 10);
    const auto base = fs::file_time_type::clock::now() - std::chrono::hours(2);
    fs::last_write_time(old, base);
    fs::last_write_time(recent, base + std::chrono::minutes(1));
    auto budget = Budget::configure(root, {}, spaceWith(
        std::make_shared<std::atomic<std::uint64_t>>(900)));
    budget->waitForIdle();

    auto pin = budget->pinRead(old);
    pin.complete(true);
    budget->updateLimits({10, 0});
    budget->waitForIdle();
    CHECK(fs::exists(old));
    CHECK_FALSE(fs::exists(recent));
    CHECK(budget->stats().managedBytes == 10);
    fs::remove_all(root);
}

TEST_CASE("lowering a maximum starts an immediate background trim")
{
    const auto root = tempRoot("trim");
    writeBytes(volumeChunk(root, 0), 20);
    writeBytes(volumeChunk(root, 1), 20);
    auto budget = Budget::configure(root, {}, spaceWith(
        std::make_shared<std::atomic<std::uint64_t>>(900)));
    budget->waitForIdle();
    budget->updateLimits({15, 0});
    budget->waitForIdle();
    CHECK(budget->stats().managedBytes <= 15);
    fs::remove_all(root);
}

TEST_CASE("unlimited maximum still enforces the free-space reserve")
{
    const auto root = tempRoot("reserve");
    writeBytes(volumeChunk(root, 0), 20);
    auto free = std::make_shared<std::atomic<std::uint64_t>>(55);
    auto budget = Budget::configure(root, {std::nullopt, 50}, spaceWith(free));
    budget->waitForIdle();
    CHECK(publish(*budget, volumeChunk(root, 1), 10));
    CHECK_FALSE(fs::exists(volumeChunk(root, 0)));
    CHECK(budget->stats().managedBytes == 10);
    fs::remove_all(root);
}

TEST_CASE("below reserve blocks growth but permits equal-size replacement")
{
    const auto root = tempRoot("below");
    const auto target = volumeChunk(root, 0);
    writeBytes(target, 10, 'a');
    auto free = std::make_shared<std::atomic<std::uint64_t>>(40);
    auto budget = Budget::configure(root, {std::nullopt, 50}, spaceWith(free));
    budget->waitForIdle();

    auto replacement = budget->reserveWrite(target, 10);
    REQUIRE(static_cast<bool>(replacement));
    writeBytes(target, 10, 'b');
    replacement.commit();
    auto growth = budget->reserveWrite(volumeChunk(root, 1), 1);
    REQUIRE(static_cast<bool>(growth));
    CHECK_FALSE(fs::exists(target));
    growth.cancel();
    CHECK_FALSE(static_cast<bool>(budget->reserveWrite(volumeChunk(root, 2), 1)));
    CHECK(budget->stats().lowSpace);
    fs::remove_all(root);
}

TEST_CASE("oversized writes and concurrent reservations cannot exceed a maximum")
{
    const auto root = tempRoot("concurrent");
    auto budget = Budget::configure(root, {100, 0}, spaceWith(
        std::make_shared<std::atomic<std::uint64_t>>(900)));
    budget->waitForIdle();
    CHECK_FALSE(static_cast<bool>(budget->reserveWrite(volumeChunk(root, 0), 101)));
    auto first = budget->reserveWrite(volumeChunk(root, 0), 60);
    REQUIRE(static_cast<bool>(first));
    CHECK_FALSE(static_cast<bool>(budget->reserveWrite(volumeChunk(root, 1), 50)));
    first.cancel();
    CHECK(static_cast<bool>(budget->reserveWrite(volumeChunk(root, 1), 50)));
    fs::remove_all(root);
}

TEST_CASE("cancelled and failed writes do not change accounting")
{
    const auto root = tempRoot("failure");
    auto budget = Budget::configure(root, {100, 0}, spaceWith(
        std::make_shared<std::atomic<std::uint64_t>>(900)));
    budget->waitForIdle();
    auto reservation = budget->reserveWrite(volumeChunk(root, 0), 40);
    REQUIRE(static_cast<bool>(reservation));
    reservation.cancel();
    CHECK(budget->stats().managedBytes == 0);
    CHECK(static_cast<bool>(budget->reserveWrite(volumeChunk(root, 1), 100)));
    fs::remove_all(root);
}
