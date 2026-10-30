// Zarr arrays read over sftp:// through the production transport (see sftp_fixture.hpp).
#include "vc_reader/zarr.hpp"
#include "vc_reader/native/array.hpp"
#include "vc_reader/sftp.hpp"
#include "vc_reader/native/http.hpp"
#include "sftp_fixture.hpp"

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {
void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void write_file(const fs::path& path, const std::vector<uint8_t>& bytes)
{
    fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
}

void write_text(const fs::path& path, const std::string& text)
{
    fs::create_directories(path.parent_path());
    std::ofstream(path) << text;
}


// 4^3 uint8 array of 2^3 chunks; chunk (1,1,1) is absent and reads as fill 5.
std::vector<uint8_t> volume()
{
    std::vector<uint8_t> v(64);
    for (size_t i = 0; i < v.size(); ++i)
        v[i] = uint8_t(10 + i);
    return v;
}

std::vector<uint8_t> chunk_of(const std::vector<uint8_t>& v, int cz, int cy, int cx)
{
    std::vector<uint8_t> out;
    for (int z = 0; z < 2; ++z)
        for (int y = 0; y < 2; ++y)
            for (int x = 0; x < 2; ++x)
                out.push_back(v[((cz * 2 + z) * 4 + cy * 2 + y) * 4 + cx * 2 + x]);
    return out;
}

void check(const std::string& location, const std::vector<uint8_t>& v, const std::string& what)
{
    auto reader = vcr::zarr::openPyramidLevel(location, 0, {});
    require(reader.meta.shape == vcr::Index{4, 4, 4}, what + ": shape");
    std::vector<uint8_t> out(64);
    reader.array->read_many({{0, 0, 0}}, {{4, 4, 4}}, {out.data()});
    for (int z = 0; z < 4; ++z)
        for (int y = 0; y < 4; ++y)
            for (int x = 0; x < 4; ++x) {
                const size_t i = (z * 4 + y) * 4 + x;
                const uint8_t want = z >= 2 && y >= 2 && x >= 2 ? 5 : v[i];
                require(out[i] == want, what + ": voxel " + std::to_string(i));
            }
}
}  // namespace

int main(int argc, char** argv)
{
    if (const int status = sftp_fixture::run_as_ssh(argc, argv); status >= 0)
        return status;
    const auto root = fs::temp_directory_path() /
                      ("vc-reader-sftp-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Cleanup {
        fs::path path;
        ~Cleanup()
        {
            std::error_code ec;
            fs::remove_all(path, ec);
        }
    } cleanup{root};
    try {
        if (!sftp_fixture::install(root)) {
            std::cout << "test_native_sftp: no installed OpenSSH sftp-server; skipped\n";
            return 77;
        }
        const auto v = volume();
        // Zarr v2, raw chunks.
        const auto v2 = root / "v2.zarr";
        write_text(v2 / ".zarray", R"({"zarr_format":2,"shape":[4,4,4],"chunks":[2,2,2],"dtype":"|u1",)"
                                   R"("compressor":null,"filters":null,"fill_value":5,"order":"C"})");
        for (int cz = 0; cz < 2; ++cz)
            for (int cy = 0; cy < 2; ++cy)
                for (int cx = 0; cx < 2; ++cx)
                    if (cz + cy + cx < 3)
                        write_file(v2 / (std::to_string(cz) + "." + std::to_string(cy) + "." + std::to_string(cx)),
                                   chunk_of(v, cz, cy, cx));
        check(sftp_fixture::url(v2), v, "v2");
        check("ssh://test-host" + vcr::sftp::encodePath(v2.string()), v, "ssh:// alias");

        // Zarr v3, one shard of all eight chunks: index and chunks are byte ranges.
        const auto v3 = root / "v3 sharded.zarr";
        write_text(v3 / "zarr.json",
                   R"({"zarr_format":3,"node_type":"array","shape":[4,4,4],"data_type":"uint8",)"
                   R"("chunk_grid":{"name":"regular","configuration":{"chunk_shape":[4,4,4]}},)"
                   R"("chunk_key_encoding":{"name":"default","configuration":{"separator":"/"}},"fill_value":5,)"
                   R"("codecs":[{"name":"sharding_indexed","configuration":{"chunk_shape":[2,2,2],)"
                   R"("codecs":[{"name":"bytes","configuration":{"endian":"little"}}],)"
                   R"("index_codecs":[{"name":"bytes","configuration":{"endian":"little"}}],"index_location":"end"}}]})");
        std::vector<uint8_t> shard;
        std::vector<uint64_t> index;
        for (int cz = 0; cz < 2; ++cz)
            for (int cy = 0; cy < 2; ++cy)
                for (int cx = 0; cx < 2; ++cx) {
                    if (cz + cy + cx == 3) {
                        index.push_back(~uint64_t(0));
                        index.push_back(~uint64_t(0));
                        continue;
                    }
                    const auto c = chunk_of(v, cz, cy, cx);
                    index.push_back(shard.size());
                    index.push_back(c.size());
                    shard.insert(shard.end(), c.begin(), c.end());
                }
        const auto* p = reinterpret_cast<const uint8_t*>(index.data());
        shard.insert(shard.end(), p, p + index.size() * sizeof(uint64_t));
        write_file(v3 / "c/0/0/0", shard);
        check(sftp_fixture::url(v3), v, "v3 sharded");

        // The transport itself: ranges, absent objects, listing.
        vcr::HttpClient http(vcr::HttpOptions{});
        std::vector<uint8_t> body;
        require(!http.get(sftp_fixture::url(v2 / "1.1.1"), body), "absent object reads as not found");
        const auto ranged = http.get_range(sftp_fixture::url(v3 / "c/0/0/0"), body, "-16", 16);
        require(ranged.status == 206 && body.size() == 16 && ranged.content_range == "bytes " +
                    std::to_string(shard.size() - 16) + "-" + std::to_string(shard.size() - 1) + "/" +
                    std::to_string(shard.size()), "suffix range");
        const auto listed = vcr::sftp::Fetcher().list(sftp_fixture::url(root));
        require(listed.size() == 3 && listed[0].directory && listed[0].name == "bin", "directory listing");
        std::cout << "sftp:// v2, ssh:// alias, sharded v3, ranges and listing passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
