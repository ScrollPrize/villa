#include "TestSupport.hpp"
#include "vc_object_store/PackedTrackStore.hpp"

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

namespace {
std::int64_t fileMtimeNs(const fs::path& path)
{
    const auto modified = std::chrono::file_clock::to_sys(fs::last_write_time(path));
    return std::chrono::duration_cast<std::chrono::nanoseconds>(modified.time_since_epoch()).count();
}

void setFileMtimeNs(const fs::path& path, std::int64_t nanoseconds)
{
    const std::chrono::sys_time<std::chrono::nanoseconds> stamp{std::chrono::nanoseconds{nanoseconds}};
    fs::last_write_time(path, std::chrono::file_clock::from_sys(stamp));
    require(fileMtimeNs(path) == nanoseconds, "filesystem did not keep the requested mtime");
}

struct Fixture {
    fs::path dbm;
    fs::path packed;
};

nlohmann::json signatureEntry(const nlohmann::json& size, const nlohmann::json& mtime)
{
    nlohmann::json entry = nlohmann::json::array();
    entry.push_back("tracks.dbm");
    entry.push_back(size);
    entry.push_back(mtime);
    nlohmann::json signature = nlohmann::json::array();
    signature.push_back(std::move(entry));
    return signature;
}

Fixture prepare(const fs::path& root, std::int64_t fileSize, std::int64_t fileMtime, const nlohmann::json& signature)
{
    const auto dbm = root / "tracks.dbm";
    const auto packed = root / "tracks.dbm.vctracks";
    fs::remove_all(packed);
    {
        std::ofstream out(dbm, std::ios::binary | std::ios::trunc);
        const std::string bytes(static_cast<std::size_t>(fileSize), 'x');
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        require(static_cast<bool>(out), "write dbm");
    }
    setFileMtimeNs(dbm, fileMtime);
    fs::create_directory(packed);
    nlohmann::json metadata = nlohmann::json::object();
    metadata["version"] = 1;
    metadata["source_db_signature"] = signature;
    std::ofstream meta(packed / "metadata.json");
    meta << metadata.dump(2);
    require(static_cast<bool>(meta), "write metadata");
    return {dbm, packed};
}

void accept(const Fixture& fixture, const char* label)
{
    try {
        const auto resolved = vc::resolvePackedTrackStore(fixture.dbm);
        require(resolved == fs::canonical(fixture.packed), label);
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string(label) + ": " + error.what());
    }
}

void reject(const Fixture& fixture, const char* label)
{
    try {
        vc::resolvePackedTrackStore(fixture.dbm);
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(std::string(label) + " threw a JSON type error: " + error.what());
    } catch (const std::exception& error) {
        require(std::string(error.what()).find("packed tracks are stale") != std::string::npos, label);
        return;
    }
    throw std::runtime_error(std::string(label) + " was accepted");
}
}

int main()
try {
    // A real DBM mtime_ns, kept off a whole second and aligned to 100 ns so
    // Windows file times (100 ns units) can store it. Mirrors and HTTP
    // Last-Modified keep only the second.
    constexpr std::int64_t kNanoseconds = 1785295357776482700LL;
    constexpr std::int64_t kSecond = 1000000000LL;
    const std::int64_t truncated = kNanoseconds / kSecond * kSecond;
    TemporaryDirectory temp;

    accept(prepare(temp.path, 10, kNanoseconds, signatureEntry(10, kNanoseconds)), "exact match");
    accept(prepare(temp.path, 10, truncated, signatureEntry(10, kNanoseconds)), "whole-second copy");
    accept(prepare(temp.path, 10, kNanoseconds, signatureEntry(10, truncated)), "whole-second stored stamp");
    reject(prepare(temp.path, 10, kNanoseconds + kSecond, signatureEntry(10, kNanoseconds)), "different second");
    reject(prepare(temp.path, 10, truncated + kSecond, signatureEntry(10, kNanoseconds)), "different second from a whole-second copy");
    reject(prepare(temp.path, 10, kNanoseconds + 1000, signatureEntry(10, kNanoseconds)), "nanoseconds inside the same second");
    reject(prepare(temp.path, 11, kNanoseconds, signatureEntry(10, kNanoseconds)), "size mismatch");
    reject(prepare(temp.path, 10, kNanoseconds, signatureEntry(nullptr, kNanoseconds)), "malformed size");
    reject(prepare(temp.path, 10, kNanoseconds, nullptr), "malformed signature");

    std::cout << "packed track signature tests passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
