// Bounded single-thread codec/read benchmark. Optional arguments are existing
// Blosc chunk files with layout [7,64,64,64] uint8; no inputs are modified.
#include <vc_reader/native/channel_decode.hpp>
#include <vc_reader/native/chunk_io.hpp>
#include <chrono>
#include <iostream>

using Clock = std::chrono::steady_clock;
constexpr std::size_t plane = 64*64*64, total = 7*plane;

template<class Read>
void measure(const std::string& name, vcr::ChunkCodec codec, std::size_t encodedSize,
             Read&& read, const std::vector<std::uint8_t>& reference, Clock::time_point deadline)
{
    const std::vector<std::size_t> channels{4,5,6};
    for (bool partial : {false,true}) {
        std::vector<double> times;
        std::size_t readBytes = 0, decodedBytes = 0;
        for (int iteration = 0; iteration < 30 && Clock::now() < deadline; ++iteration) {
            const auto begin = Clock::now();
            if (partial) {
                auto result = vcr::decode_channels(codec,total,plane,channels,channels,encodedSize,read);
                times.push_back(std::chrono::duration<double,std::milli>(Clock::now()-begin).count());
                readBytes = result.bytesRead; decodedBytes = result.bytesDecoded;
                for (auto& channel : result.channels)
                    if (!std::equal(channel.bytes->begin(),channel.bytes->end(),reference.begin()+channel.index*plane))
                        throw std::runtime_error("partial output mismatch");
            } else {
                std::vector<std::uint8_t> encoded(encodedSize),decoded(total);
                read(0,encoded.size(),encoded.data());
                vcr::decode(codec,encoded.data(),encoded.size(),decoded.data(),decoded.size());
                times.push_back(std::chrono::duration<double,std::milli>(Clock::now()-begin).count());
                readBytes = encodedSize; decodedBytes = total;
                if (decoded != reference) throw std::runtime_error("full output mismatch");
            }
        }
        if (times.empty()) continue;
        const auto mean = std::accumulate(times.begin(),times.end(),0.0)/times.size();
        std::sort(times.begin(),times.end());
        std::cout << name << ',' << (partial ? "channels456" : "full") << ',' << times.size() << ','
                  << mean << ',' << times[times.size()/2] << ',' << times[(times.size()-1)*95/100] << ','
                  << times.front() << ',' << times.back() << ',' << readBytes << ',' << decodedBytes << '\n';
    }
}

int main(int argc,char** argv)
{
    try {
        const auto deadline = Clock::now()+std::chrono::seconds(25);
        std::cout << "input,mode,repetitions,mean_ms,p50_ms,p95_ms,min_ms,max_ms,read_bytes,decoded_bytes\n";
        if (argc > 1) {
            for (int i = 1; i < argc && Clock::now() < deadline; ++i) {
                auto file = vcr::File::open_read(argv[i]);
                if (!file) throw std::runtime_error("missing input");
                auto encoded = vcr::read_whole(file.fd(),argv[i]);
                std::vector<std::uint8_t> reference(total);
                vcr::decode(vcr::Codec::Blosc,encoded.data(),encoded.size(),reference.data(),reference.size());
                measure(argv[i],{vcr::Codec::Blosc},encoded.size(),
                    [&](auto offset,auto length,auto dst) { vcr::pread_exact(file.fd(),dst,length,offset,argv[i]); },
                    reference,deadline);
            }
        } else {
            std::vector<std::uint8_t> reference(total);
            std::uint32_t random = 42;
            for (std::size_t i = 0; i < total; ++i) {
                random ^= random << 13; random ^= random >> 17; random ^= random << 5;
                reference[i] = std::uint8_t((i/64+i/plane*31+(random&15))%256);
            }
            for (auto codec : {vcr::Codec::Raw,vcr::Codec::Blosc}) {
                vcr::EncodeOptions options; options.cname="zstd"; options.level=3;
                const auto encoded = vcr::encode(codec,options,reference.data(),reference.size(),1,{7,64,64,64});
                measure(codec == vcr::Codec::Raw ? "synthetic_raw" : "synthetic_blosc",{codec},encoded.size(),
                    [&](auto offset,auto size,auto dst) { std::memcpy(dst,encoded.data()+offset,size); },reference,deadline);
            }
        }
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
