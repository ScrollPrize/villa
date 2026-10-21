#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/render/PersistentZarrCacheBudget.hpp"
#include "vc/core/render/detail/CacheAgeBuckets.hpp"

#include <atomic>
#include <chrono>
#include <fstream>
#include <random>
#include <thread>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <future>
#include <map>

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

TEST_CASE("disk budget synthetic eviction benchmark")
{
    if (!std::getenv("VC_CACHE_BUDGET_BENCH"))
        return;
    constexpr int count = 20000;
    const auto root = tempRoot("benchmark");
    for (int i = 0; i < count; ++i)
        writeBytes(volumeChunk(root, i), 100);
    auto budget = Budget::configure(root, {}, spaceWith(
        std::make_shared<std::atomic<std::uint64_t>>(1000000000)));
    budget->waitForIdle();
    const auto start = std::chrono::steady_clock::now();
    const auto cpu = std::clock();
    budget->updateLimits({count * 50, 0});
    budget->waitForIdle();
    const auto trimmed = std::chrono::steady_clock::now();
    for (int i = count; i < count + 200; ++i)
        REQUIRE(publish(*budget, volumeChunk(root, i), 100));
    budget->waitForIdle();
    const auto finished = std::chrono::steady_clock::now();
    std::cout << "budget_bench entries=" << count
              << " trim_s=" << std::chrono::duration<double>(trimmed - start).count()
              << " churn_s=" << std::chrono::duration<double>(finished - trimmed).count()
              << " cpu_s=" << double(std::clock() - cpu) / CLOCKS_PER_SEC << '\n';
    fs::remove_all(root);
}

TEST_CASE("generation buckets lazily refresh age and survive wrap and long idle gaps")
{
    using Ages = vc::render::detail::CacheAgeBuckets<int>;
    Ages ages(100);
    const auto old = ages.insert(1, 70);
    const auto middle = ages.insert(2, 90);
    const auto young = ages.insert(3, 99);
    ages.touch(old, 100);
    CHECK(ages.takeOldest() == middle);
    ages.erase(middle);
    ages.suspend(young);
    CHECK(ages.takeOldest() == old);
    ages.resume(old);
    ages.advance(132);
    ages.resume(young);
    const auto newest = ages.insert(4, 132);
    const auto first = ages.takeOldest();
    CHECK(first != newest);
    ages.erase(first);
    const auto second = ages.takeOldest();
    CHECK(second != newest);
    ages.erase(second);
    ages.advance(10000);
    CHECK(ages.takeOldest() == newest);
    ages.erase(newest);
    CHECK(ages.empty());
    const auto reused = ages.insert(5, 10000);
    CHECK(ages.value(ages.takeOldest()) == 5);
    ages.erase(reused);
}

TEST_CASE("generation bucket lazy work is bounded and pinned slots survive recycling")
{
    using Ages = vc::render::detail::CacheAgeBuckets<int>;
    Ages ages(100);
    std::vector<Ages::Handle> ids;
    for (int i = 0; i < 1000; ++i) {
        ids.push_back(ages.insert(i, 80));
        ages.touch(ids.back(), 100);
    }
    CHECK(ages.takeOldest(8) == Ages::none);
    CHECK_FALSE(ages.empty());
    ages.suspend(ids.back());
    ages.advance(1000);
    ages.touch(ids.back(), 1000);
    ages.resume(ids.back());
    for (int i = 0; i < 999; ++i) {
        const auto id = ages.takeOldest();
        REQUIRE(id != Ages::none);
        CHECK(id != ids.back());
        ages.erase(id);
    }
    CHECK(ages.takeOldest() == ids.back());
}

TEST_CASE("generation buckets match a reference age model under mixed operations")
{
    using Ages = vc::render::detail::CacheAgeBuckets<int>;
    std::uint64_t now = 100;
    Ages ages(now);
    struct Reference { std::uint64_t generation; bool pinned; };
    std::map<Ages::Handle, Reference> reference;
    std::mt19937 random(17);
    auto ageClass = [&](std::uint64_t gen) { return std::max(gen, now - 32); };
    for (int step = 0; step < 10000; ++step) {
        const auto op = random() % 6;
        if (op == 0 || reference.empty()) {
            reference.emplace(ages.insert(step, now), Reference{now, false});
        } else if (op == 1) {
            now += random() % 70;
            ages.advance(now);
        } else if (op == 2) {
            const auto id = ages.takeOldest(10001);
            std::uint64_t expected = now + 1;
            for (const auto& [key, entry] : reference)
                if (!entry.pinned) expected = std::min(expected, ageClass(entry.generation));
            if (expected == now + 1) {
                CHECK(id == Ages::none);
            } else {
                REQUIRE(id != Ages::none);
                CHECK_FALSE(reference.at(id).pinned);
                CHECK(ageClass(reference.at(id).generation) == expected);
                reference.erase(id);
                ages.erase(id);
            }
        } else {
            auto it = reference.begin();
            std::advance(it, random() % reference.size());
            if (op == 3) {
                it->second.generation = now;
                ages.touch(it->first, now);
            } else if (op == 4) {
                it->second.pinned = !it->second.pinned;
                if (it->second.pinned) ages.suspend(it->first);
                else ages.resume(it->first);
            } else {
                ages.erase(it->first);
                reference.erase(it);
            }
        }
    }
}

TEST_CASE("slow vacuum space query does not block readers writers or statistics")
{
    const auto root = tempRoot("async");
    std::atomic<bool> armed{false};
    std::promise<void> entered, release;
    auto enteredFuture = entered.get_future();
    auto releaseFuture = release.get_future().share();
    auto budget = Budget::configure(root, {}, [&](const fs::path&, std::error_code& ec) {
        ec.clear();
        if (armed.exchange(false)) {
            entered.set_value();
            releaseFuture.wait();
        }
        return fs::space_info{100000, 90000, 90000};
    });
    budget->waitForIdle();
    CHECK(publish(*budget, volumeChunk(root, 0), 100));
    budget->waitForIdle();
    armed = true;
    budget->updateLimits({50, 0});
    const bool enteredInTime = enteredFuture.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
    CHECK(enteredInTime);
    auto work = std::async(std::launch::async, [&] {
        auto pin = budget->pinRead(volumeChunk(root, 0));
        CHECK(budget->stats().managedBytes == 100);
        const bool written = publish(*budget, volumeChunk(root, 1), 10);
        pin.complete();
        return written;
    });
    const auto state = work.wait_for(std::chrono::seconds(5));
    release.set_value(); // Unblock even if the assertion failed.
    CHECK(state == std::future_status::ready);
    CHECK(work.get());
    budget->waitForIdle();
    CHECK(budget->stats().managedBytes <= 50);
    fs::remove_all(root);
}

TEST_CASE("pinned cache survives pressure and vacuum retries after unpin")
{
    const auto root = tempRoot("pinned");
    auto budget = Budget::configure(root, {}, spaceWith(
        std::make_shared<std::atomic<std::uint64_t>>(900)));
    budget->waitForIdle();
    CHECK(publish(*budget, volumeChunk(root, 0), 50));
    auto pin = budget->pinRead(volumeChunk(root, 0));
    budget->updateLimits({0, 0});
    budget->waitForIdle();
    CHECK(fs::exists(volumeChunk(root, 0)));
    CHECK(budget->stats().managedBytes == 50);
    pin.complete();
    budget->waitForIdle();
    CHECK_FALSE(fs::exists(volumeChunk(root, 0)));
    CHECK(budget->stats().managedBytes == 0);
    fs::remove_all(root);
}

TEST_CASE("deleted files and failed removals cannot cause a vacuum busy loop")
{
    const auto root = tempRoot("delete_failure");
    const auto missing = volumeChunk(root, 0), blocked = volumeChunk(root, 1);
    writeBytes(missing, 10);
    writeBytes(blocked, 10);
    auto budget = Budget::configure(root, {}, spaceWith(
        std::make_shared<std::atomic<std::uint64_t>>(900)));
    budget->waitForIdle();
    fs::remove(missing);
    fs::remove(blocked);
    writeBytes(blocked / "child", 1); // Nonempty directory: portable remove failure.
    budget->updateLimits({0, 0});
    budget->waitForIdle();
    CHECK(budget->stats().managedBytes == 10);
    fs::remove_all(blocked);
    budget->updateLimits({0, 0});
    budget->waitForIdle();
    CHECK(budget->stats().managedBytes == 0);
    fs::remove_all(root);
}

TEST_CASE("ordinary subtree operations do not inventory unrelated new files")
{
    const auto root = tempRoot("no_rescan");
    writeBytes(volumeChunk(root, 0), 10);
    auto budget = Budget::configure(root, {}, spaceWith(
        std::make_shared<std::atomic<std::uint64_t>>(900)));
    budget->waitForIdle();
    writeBytes(root / "other" / "level_0" / "0" / "0" / "0.bin", 20);
    std::error_code ec;
    CHECK(budget->moveCacheSubtree(root / "volume-hash", root / "moved", ec));
    budget->waitForIdle();
    CHECK(budget->stats().managedBytes == 10);
    CHECK(budget->removeCacheSubtree(root / "moved", ec));
    budget->waitForIdle();
    CHECK(budget->stats().managedBytes == 0);
    fs::remove_all(root);
}

TEST_CASE("writes concurrent with startup inventory are merged exactly once")
{
    const auto root = tempRoot("startup_writes");
    for (int i = 0; i < 1000; ++i)
        writeBytes(volumeChunk(root, i), 1);
    auto budget = Budget::configure(root, {}, spaceWith(
        std::make_shared<std::atomic<std::uint64_t>>(100000)));
    for (int i = 0; i < 100; ++i) {
        CHECK(publish(*budget, volumeChunk(root, i), 3));
        CHECK(publish(*budget, volumeChunk(root, 1000 + i), 5));
    }
    auto cancelled = budget->reserveWrite(volumeChunk(root, 500), 999);
    cancelled.cancel();
    budget->waitForIdle();
    CHECK(budget->stats().managedBytes == 1700);
    fs::remove_all(root);
}

TEST_CASE("idle status and reads do not poll the filesystem free space")
{
    const auto root = tempRoot("no_poll");
    std::atomic<int> calls{0};
    auto budget = Budget::configure(root, {}, [&](const fs::path&, std::error_code& ec) {
        ++calls;
        ec.clear();
        return fs::space_info{1000, 900, 900};
    });
    budget->waitForIdle();
    CHECK(publish(*budget, volumeChunk(root, 0), 10));
    budget->waitForIdle();
    const auto before = calls.load();
    for (int i = 0; i < 100; ++i) {
        auto pin = budget->pinRead(volumeChunk(root, 0));
        pin.complete();
        CHECK(budget->stats().managedBytes == 10);
    }
    budget->waitForIdle();
    CHECK(calls.load() == before);
    fs::remove_all(root);
}

TEST_CASE("moving a subtree replaces stale destination accounting")
{
    const auto root = tempRoot("move_stale");
    writeBytes(volumeChunk(root, 0), 10);
    writeBytes(root / "moved" / "level_0" / "0" / "0" / "0.bin", 20);
    auto budget = Budget::configure(root, {}, spaceWith(
        std::make_shared<std::atomic<std::uint64_t>>(900)));
    budget->waitForIdle();
    fs::remove_all(root / "moved");
    std::error_code ec;
    CHECK(budget->moveCacheSubtree(root / "volume-hash", root / "moved", ec));
    budget->waitForIdle();
    CHECK(budget->stats().managedBytes == 10);
    budget->updateLimits({0, 0});
    budget->waitForIdle();
    CHECK(budget->stats().managedBytes == 0);
    fs::remove_all(root);
}

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

TEST_CASE("budget startup scan discovers Delta3D Zarr chunks")
{
    const auto root = tempRoot("delta3d_discover");
    const auto volume = root / "volume-hash";
    fs::create_directories(volume / "0");
    {
        std::ofstream metadata(volume / "0" / ".zarray");
        metadata << R"({"zarr_format":2,"shape":[8,8,8],"chunks":[2,2,2],"dtype":"|u1","compressor":{"id":"vc-delta3d","quant":1},"fill_value":0,"order":"C","filters":null})";
    }
    writeBytes(volume / "0" / "0.0.0", 23);
    writeBytes(root / "volume-hash" / ".vc_delta3d_cache", 5);

    auto budget = Budget::configure(root, {}, spaceWith(
        std::make_shared<std::atomic<std::uint64_t>>(900)));
    budget->waitForIdle();
    CHECK(budget->stats().managedBytes == 23);
    fs::remove_all(root);
}

TEST_CASE("budget discovers open-data volume chunks grouped by sample")
{
    const auto root = tempRoot("open_data_volume");
    writeBytes(root / "open_data" / "volumes" / "PHerc0139" /
                   "volume-hash" / "level_0" / "0" / "0" / "0.bin",
               29);

    auto budget = Budget::configure(root, {}, spaceWith(
        std::make_shared<std::atomic<std::uint64_t>>(900)));
    budget->waitForIdle();
    CHECK(budget->stats().managedBytes == 29);
    fs::remove_all(root);
}

TEST_CASE("budget-coordinated cache migration preserves accounting")
{
    const auto root = tempRoot("move_subtree");
    const auto legacyVolume = root / "volume-hash";
    const auto legacyChunk =
        legacyVolume / "level_0" / "0" / "0" / "0.bin";
    const auto groupedVolume =
        root / "open_data" / "volumes" / "PHerc0139" / "volume-hash";
    writeBytes(legacyChunk, 37);

    auto budget = Budget::configure(root, {}, spaceWith(
        std::make_shared<std::atomic<std::uint64_t>>(900)));
    budget->waitForIdle();
    REQUIRE(budget->stats().managedBytes == 37);

    std::error_code ec;
    CHECK(budget->moveCacheSubtree(legacyVolume, groupedVolume, ec));
    CHECK_FALSE(ec);
    CHECK_FALSE(fs::exists(legacyVolume));
    CHECK(fs::is_regular_file(
        groupedVolume / "level_0" / "0" / "0" / "0.bin"));
    CHECK(budget->stats().managedBytes == 37);
    budget->waitForIdle();
    CHECK(budget->stats().managedBytes == 37);

    auto pin = budget->pinRead(
        groupedVolume / "level_0" / "0" / "0" / "0.bin");
    pin.complete(false);
    fs::remove_all(root);
}

TEST_CASE("budget-coordinated cache replacement removes stale subtree accounting")
{
    const auto root = tempRoot("replace_subtree");
    const auto oldVolume = root / "old-volume";
    const auto oldChunk = oldVolume / "level_0" / "0" / "0" / "0.bin";
    writeBytes(oldChunk, 100);
    auto budget = Budget::configure(root, {101, 0}, spaceWith(
        std::make_shared<std::atomic<std::uint64_t>>(900)));
    budget->waitForIdle();
    REQUIRE(budget->stats().managedBytes == 100);

    std::error_code ec;
    CHECK(budget->removeCacheSubtree(oldVolume, ec));
    CHECK_FALSE(ec);
    CHECK_FALSE(fs::exists(oldVolume));
    budget->waitForIdle();
    CHECK(budget->stats().managedBytes == 0);
    CHECK_FALSE(fs::exists(root / ".vc_cache_retired"));

    CHECK(publish(*budget, volumeChunk(root, 1), 100));
    CHECK(budget->stats().managedBytes == 100);
    fs::remove_all(root);
}

TEST_CASE("budget discovers native Zarr payloads but excludes metadata")
{
    const auto root = tempRoot("native_zarr");
    const auto mirror = root / "volume-hash";
    writeBytes(mirror / ".zgroup", 17);
    writeBytes(mirror / ".zattrs", 19);
    fs::create_directories(mirror / "scale0");
    {
        std::ofstream metadata(mirror / "scale0" / ".zarray");
        metadata << R"({"zarr_format":2,"shape":[8,8,8],"chunks":[2,2,2],"dtype":"|u1","compressor":null,"fill_value":0,"order":"C","filters":null})";
    }
    writeBytes(mirror / "scale0" / ".zattrs", 29);
    writeBytes(mirror / "scale0" / "0.0.0", 31);
    writeBytes(mirror / "scale0" / "0.0.1.empty", 0);
    writeBytes(mirror / "unrelated.txt", 37);
    writeBytes(mirror / "scale0" / "unrelated.txt", 43);

    auto budget = Budget::configure(root, {}, spaceWith(
        std::make_shared<std::atomic<std::uint64_t>>(900)));
    budget->waitForIdle();
    CHECK(budget->stats().managedBytes == 31);
    fs::remove_all(root);
}

TEST_CASE("budget discovers native Zarr v3 shard payloads")
{
    const auto root = tempRoot("native_zarr_v3");
    const auto mirror = root / "volume-hash";
    writeBytes(mirror / "zarr.json", 17);
    fs::create_directories(mirror / "scale0");
    {
        std::ofstream metadata(mirror / "scale0" / "zarr.json");
        metadata << R"({"zarr_format":3,"node_type":"array","shape":[8,8,8]})";
    }
    writeBytes(mirror / "scale0" / "c" / "0" / "0" / "0", 41);
    writeBytes(mirror / "scale0" / "c" / "bad" / "0" / "0", 47);

    auto budget = Budget::configure(root, {}, spaceWith(
        std::make_shared<std::atomic<std::uint64_t>>(900)));
    budget->waitForIdle();
    CHECK(budget->stats().managedBytes == 41);
    fs::remove_all(root);
}

TEST_CASE("protected metadata remains writable under disk pressure and is never evicted")
{
    const auto root = tempRoot("protected_metadata");
    const auto metadata = root / "volume-hash" / "zarr.json";
    auto free = std::make_shared<std::atomic<std::uint64_t>>(40);
    auto budget = Budget::configure(root, {1, 50}, spaceWith(free));
    budget->waitForIdle();
    auto reservation = budget->reserveProtectedWrite(metadata, 10);
    REQUIRE(static_cast<bool>(reservation));
    writeBytes(metadata, 10);
    reservation.commit();
    budget->waitForIdle();
    CHECK(fs::is_regular_file(metadata));
    CHECK(budget->stats().managedBytes == 0);
    CHECK(budget->stats().lowSpace);
    budget->updateLimits({0, 0});
    budget->waitForIdle();
    CHECK(fs::is_regular_file(metadata));
    fs::remove_all(root);
}

TEST_CASE("volume chunk exclusions are relative to the managed root")
{
    const auto base = tempRoot("relative");
    const auto root = base / "projects" / "user-cache";
    writeBytes(volumeChunk(root, 0), 17);
    auto budget = Budget::configure(root, {}, spaceWith(
        std::make_shared<std::atomic<std::uint64_t>>(900)));
    budget->waitForIdle();
    CHECK(budget->stats().managedBytes == 17);
    fs::remove_all(base);
}

TEST_CASE("nested cache roots share one budget")
{
    Budget::resetRegistryForTesting();
    const auto root = tempRoot("nested");
    const auto nested = root / "project" / "volume";
    auto free = std::make_shared<std::atomic<std::uint64_t>>(900);
    auto parentBudget = Budget::configure(root, {}, spaceWith(free));
    auto nestedBudget = Budget::configure(nested, {}, spaceWith(free));
    CHECK(parentBudget == nestedBudget);
    CHECK(Budget::findForPath(volumeChunk(nested, 0)) == parentBudget);
    parentBudget->waitForIdle();
    fs::remove_all(root);
    Budget::resetRegistryForTesting();
}

TEST_CASE("a containing root cannot be registered after its nested root")
{
    Budget::resetRegistryForTesting();
    const auto root = tempRoot("nested_rejected");
    const auto nested = root / "project" / "volume";
    auto free = std::make_shared<std::atomic<std::uint64_t>>(900);
    auto nestedBudget = Budget::configure(nested, {}, spaceWith(free));
    CHECK_THROWS(Budget::configure(root, {}, spaceWith(free)));
    nestedBudget->waitForIdle();
    fs::remove_all(root);
    Budget::resetRegistryForTesting();
}

TEST_CASE("successful reads refresh generations without touching filesystem timestamps")
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
    CHECK(fs::last_write_time(old) == base);
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

TEST_CASE("vacuum restores minimum free space including the existing deficit")
{
    const auto root = tempRoot("reserve");
    const auto old = volumeChunk(root, 0);
    const auto fresh = volumeChunk(root, 1);
    writeBytes(old, 20);
    fs::last_write_time(old, fs::file_time_type::clock::now() - std::chrono::hours(1));
    auto budget = Budget::configure(root, {std::nullopt, 50},
        [old, fresh](const fs::path&, std::error_code& ec) {
            ec.clear();
            std::uint64_t used = 0;
            for (const auto& path : {old, fresh}) {
                std::error_code ignored;
                const auto size = fs::file_size(path, ignored);
                if (!ignored) used += size;
            }
            return fs::space_info{1000, 75 - used, 75 - used};
        });
    budget->waitForIdle();
    CHECK(publish(*budget, fresh, 10));
    budget->waitForIdle();
    CHECK_FALSE(fs::exists(old));
    CHECK(fs::exists(fresh));
    CHECK(budget->stats().managedBytes == 10);
    CHECK(budget->stats().freeBytes >= 60);
    fs::remove_all(root);
}

TEST_CASE("concurrent and oversized reservations do not block on budget enforcement")
{
    const auto root = tempRoot("concurrent");
    auto budget = Budget::configure(root, {100, 50}, spaceWith(
        std::make_shared<std::atomic<std::uint64_t>>(55)));
    budget->waitForIdle();
    auto first = budget->reserveWrite(volumeChunk(root, 0), 101);
    auto second = budget->reserveWrite(volumeChunk(root, 1), 50);
    REQUIRE(static_cast<bool>(first));
    REQUIRE(static_cast<bool>(second));
    writeBytes(volumeChunk(root, 0), 101);
    first.commit();
    second.cancel();
    budget->waitForIdle();
    CHECK(budget->stats().managedBytes <= 100);
    CHECK_FALSE(fs::exists(volumeChunk(root, 0)));
    fs::remove_all(root);
}

TEST_CASE("reads wait for an active write reservation")
{
    const auto root = tempRoot("read_write");
    const auto target = volumeChunk(root, 0);
    auto budget = Budget::configure(root, {}, spaceWith(
        std::make_shared<std::atomic<std::uint64_t>>(900)));
    budget->waitForIdle();
    auto write = budget->reserveWrite(target, 10);
    REQUIRE(static_cast<bool>(write));

    std::atomic<bool> started{false};
    std::atomic<bool> acquired{false};
    std::thread reader([&] {
        started = true;
        auto pin = budget->pinRead(target);
        acquired = true;
    });
    while (!started.load())
        std::this_thread::yield();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    CHECK_FALSE(acquired.load());
    write.cancel();
    reader.join();
    CHECK(acquired.load());
    fs::remove_all(root);
}

TEST_CASE("format replacement commits refresh all affected accounting")
{
    const auto root = tempRoot("replacement");
    const auto raw = volumeChunk(root, 0);
    auto compressed = raw;
    compressed.replace_extension(".zst");
    writeBytes(raw, 10);
    auto budget = Budget::configure(root, {}, spaceWith(
        std::make_shared<std::atomic<std::uint64_t>>(900)));
    budget->waitForIdle();

    auto reservation = budget->reserveWrite(compressed, 7, {raw});
    REQUIRE(static_cast<bool>(reservation));
    writeBytes(compressed, 7);
    fs::remove(raw);
    reservation.commit();
    CHECK(budget->stats().managedBytes == 7);
    fs::remove_all(root);
}

TEST_CASE("failed replacement refreshes accounting after removing its target")
{
    const auto root = tempRoot("failed_replacement");
    const auto target = volumeChunk(root, 0);
    writeBytes(target, 10);
    auto budget = Budget::configure(root, {}, spaceWith(
        std::make_shared<std::atomic<std::uint64_t>>(900)));
    budget->waitForIdle();
    REQUIRE(budget->stats().managedBytes == 10);

    auto reservation = budget->reserveWrite(target, 10);
    REQUIRE(static_cast<bool>(reservation));
    fs::remove(target);
    reservation.commit();
    CHECK(budget->stats().managedBytes == 0);
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
