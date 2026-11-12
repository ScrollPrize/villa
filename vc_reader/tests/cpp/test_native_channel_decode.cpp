#include <vc_reader/native/channel_decode.hpp>
#include <iostream>
#include <random>

void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }

int main()
{
    try {
        const std::size_t plane = 10002, channels = 7;
        std::vector<std::uint8_t> plain(plane*channels);
        for (std::size_t i = 0; i < plain.size(); ++i) plain[i] = std::uint8_t((i/2)%81);
        const std::vector<std::size_t> request{4,6}, relevant{0,4,5,6};
        const auto check = [&](vcr::ChunkCodec codec, const vcr::EncodeOptions& options, bool partial) {
            auto encoded = vcr::encode(codec, options, plain.data(), plain.size(), 2, {7,5001});
            std::size_t readBytes = 0;
            auto result = vcr::decode_channels(codec, plain.size(), plane, request, relevant, encoded.size(),
                [&](std::size_t offset, std::size_t size, std::uint8_t* dst) {
                    require(offset <= encoded.size() && size <= encoded.size()-offset, "bounded read");
                    std::memcpy(dst, encoded.data()+offset, size);
                    readBytes += size;
                });
            require(result.bytesRead == readBytes, "read accounting");
            for (auto& channel : result.channels)
                require(std::equal(channel.bytes->begin(), channel.bytes->end(), plain.begin()+channel.index*plane),
                        "channel bytes equal full decode");
            require(result.partial == partial, "partial decode flag");
            if (partial) {
                require(result.bytesRead < encoded.size(), "partial physical I/O");
                require(result.bytesDecoded < plain.size(), "partial decompression");
            } else require(result.channels.size() == relevant.size(), "retain all relevant on full decode");
            return encoded;
        };
        check({vcr::Codec::Raw}, {}, true);
        for (int shuffle : {BLOSC_NOSHUFFLE, BLOSC_SHUFFLE, BLOSC_BITSHUFFLE}) {
            vcr::EncodeOptions options;
            options.cname = "zstd"; options.blocksize = 1024; options.shuffle = shuffle;
            auto encoded = check({vcr::Codec::Blosc}, options, true);
            // Reorder compressed blocks physically, maintaining their directory.
            const auto blocksize = vcr::channel_detail::u32(encoded.data()+8);
            const auto blocks = (plain.size()-1)/blocksize+1;
            std::vector<std::uint8_t> reordered(encoded.begin(), encoded.begin()+16+4*blocks);
            for (std::size_t b = blocks; b-- > 0;) {
                const auto begin = vcr::channel_detail::u32(encoded.data()+16+4*b);
                const auto end = b+1 == blocks ? encoded.size() : vcr::channel_detail::u32(encoded.data()+20+4*b);
                vcr::channel_detail::put32(reordered.data()+16+4*b, reordered.size());
                reordered.insert(reordered.end(), encoded.begin()+begin, encoded.begin()+end);
            }
            auto result = vcr::decode_channels({vcr::Codec::Blosc}, plain.size(), plane, request, relevant,
                reordered.size(), [&](auto offset, auto size, auto dst) { std::memcpy(dst, reordered.data()+offset, size); });
            for (auto& c : result.channels)
                require(std::equal(c.bytes->begin(), c.bytes->end(), plain.begin()+c.index*plane), "unordered blocks");
            // Reject invalid directory, truncated body and zero block size.
            for (int corruption = 0; corruption < 3; ++corruption) {
                auto broken = encoded;
                if (corruption == 0) vcr::channel_detail::put32(broken.data()+16, 1);
                if (corruption == 1) broken.pop_back();
                if (corruption == 2) vcr::channel_detail::put32(broken.data()+8, 0);
                bool rejected = false;
                try {
                    vcr::decode_channels({vcr::Codec::Blosc}, plain.size(), plane, request, relevant, broken.size(),
                        [&](auto offset, auto size, auto dst) { std::memcpy(dst, broken.data()+offset, size); });
                } catch (const std::runtime_error&) { rejected = true; }
                require(rejected, "reject corrupted Blosc input");
            }
        }
        vcr::EncodeOptions memcpyOptions; memcpyOptions.level = 0;
        check({vcr::Codec::Blosc}, memcpyOptions, true);
        check({vcr::Codec::Zstd}, {}, false);
        check({vcr::Codec::Gzip}, {}, false);
        // One physical block covers several relevant planes: retain them without
        // reading any more compressed bytes, but not unrelated channels.
        vcr::EncodeOptions wide; wide.blocksize = 32768;
        auto encoded = vcr::encode(vcr::Codec::Blosc, wide, plain.data(), plain.size(), 2, {7,5001});
        const std::vector<std::size_t> one{0}, tags{0,1,2,6};
        auto result = vcr::decode_channels({vcr::Codec::Blosc}, plain.size(), plane, one, tags, encoded.size(),
            [&](auto offset, auto size, auto dst) { std::memcpy(dst, encoded.data()+offset, size); });
        require(result.channels.size() >= 2 && result.channels.back().index != 6, "opportunistic relevant channels");
        std::cout << "Channel range reads: raw, Blosc shuffle/bitshuffle/memcpy/short final/unordered blocks, full codecs and corrupt inputs passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
