#include <vc_reader/native/zarr_storage.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void write(const fs::path& path, const std::vector<std::uint8_t>& bytes)
{
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
    if (!file) throw std::runtime_error("fixture write failed");
}
void put64(std::uint8_t* dst, std::uint64_t value)
{
    for (unsigned i = 0; i < 8; ++i) dst[i] = std::uint8_t(value >> (i*8));
}
int main()
{
    const auto root = fs::temp_directory_path() / ("vcr-channel-storage-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Cleanup { fs::path p; ~Cleanup() { std::error_code ec; fs::remove_all(p,ec); } } cleanup{root};
    try {
        fs::create_directories(root);
        const std::size_t plane = 4096;
        std::vector<std::uint8_t> plain(7*plane);
        for (std::size_t i = 0; i < plain.size(); ++i) plain[i] = std::uint8_t(i/99);
        vcr::EncodeOptions options; options.blocksize = 1024; options.cname = "zstd";
        const vcr::ChunkCodec codec{vcr::Codec::Blosc};
        auto encoded = vcr::encode(codec, options, plain.data(), plain.size(), 1, {7,4096});
        std::atomic<std::uint64_t> requests{0}, bytes{0};
        const std::vector<std::size_t> selected{1,5}, tags{1,3,5};
        const auto check = [&](vcr::ZarrStorage& storage, const std::string& key) {
            auto result = storage.read_channels(key, codec, plain.size(), plane, selected, tags);
            require(result && result->partial && result->bytesRead < encoded.size(), "local partial read");
            for (auto& c : result->channels)
                require(std::equal(c.bytes->begin(), c.bytes->end(), plain.begin()+c.index*plane), "storage channel equality");
            std::vector<std::uint8_t> full;
            require(storage.read(key, full) && full == encoded, "full storage read unaffected");
        };
        write(root/"0", encoded);
        vcr::ZarrStorage local(root.string(), "", ".", {}, true, false, plain.size(), nullptr, requests, bytes);
        check(local, "0");
        require(!local.read_channels("missing", codec, plain.size(), plane, selected, tags), "missing unsharded object");
        for (bool atEnd : {false,true}) {
            constexpr std::size_t indexSize = 32;
            std::vector<std::uint8_t> shard(encoded.size()+indexSize,255);
            const auto indexAt = atEnd ? encoded.size() : 0;
            const auto dataAt = atEnd ? 0 : indexSize;
            std::copy(encoded.begin(),encoded.end(),shard.begin()+dataAt);
            put64(shard.data()+indexAt, dataAt);
            put64(shard.data()+indexAt+8, encoded.size());
            write(root/"s0", shard);
            vcr::ZarrStorage storage(root.string(),"s",".",{2},atEnd,false,plain.size(),nullptr,requests,bytes);
            check(storage,"0");
            require(!storage.read_channels("1",codec,plain.size(),plane,selected,tags),"missing shard entry");
        }
        require(requests == 0 && bytes == 0,"local reads not counted as network traffic");
        std::cout << "Local channel storage and start/end shard index range reads passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
