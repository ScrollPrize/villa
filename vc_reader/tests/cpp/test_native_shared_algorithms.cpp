#include "../../src/cpp/atomic_file.hpp"
#include "../../src/cpp/element_value.hpp"
#include "../../src/cpp/memory_cache.hpp"
#include "../../src/cpp/sampling_math.hpp"
#include "../../src/cpp/transforms/shared/percentile.hpp"
#include "../../src/cpp/transforms/shared/separable_filter.hpp"
#include <vc_reader/source_reader.hpp>
#include <vc_reader/zarr.hpp>
#include <fstream>
#include <iostream>

namespace {
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<class F> void rejects(F&& fn)
{
    bool threw = false;
    try { fn(); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "invalid input was accepted");
}
void caches()
{
    vcr::Lru<int> cache(5);
    cache.put("a", 1, 2); cache.put("b", 2, 2);
    check(*cache.get("a") == 1, "cache hit");
    cache.put("c", 3, 2);
    check(!cache.get("b") && cache.bytes() == 4, "evict least recently used");
    cache.put("a", 4, 4);
    check(!cache.get("c") && *cache.get("a") == 4 && cache.bytes() == 4, "replacement cost");
    cache.put("a", 5, 6);
    check(*cache.get("a") == 4, "oversized replacement preserves cached value");
    cache.clear(); check(cache.size() == 0 && cache.bytes() == 0, "clear accounting");
    vcr::MemoryCache chunks(64,1,false);
    chunks.put("missing", nullptr);
    check(chunks.get("missing") == std::pair<bool, vcr::ChunkPtr>{true, nullptr}, "cached missing chunk");
    chunks.put("data", std::make_shared<vcr::Chunk>(64, 7));
    check(!chunks.get("missing").first && chunks.nbytes() == 64, "missing entries consume budget");
    vcr::sources::DecodedCache decoded(8, 16);
    int loads = 0;
    const auto load = [&] { ++loads; return vcr::sources::Block::allocate({8}, "|u1"); };
    { auto [block, lease] = decoded.use("a", 8, 8, load); }
    { auto [block, lease] = decoded.use("a", 8, 8, load); }
    check(loads == 1 && decoded.stats().hits == 1, "decoded cache hit and lease release");
    { auto [block, lease] = decoded.use("b", 8, 0, load); }
    { auto [block, lease] = decoded.use("a", 8, 0, load); }
    check(loads == 3, "decoded cache evicts by bytes");
}
void filters()
{
    using namespace vcr::transforms;
    const Grid grid({2, 3});
    const std::vector<float> input{1, 2, 3, 4, 5, 6};
    const std::vector<float> taps{-1, 0, 1};
    std::vector<float> out(6);
    convolve_axis(input, out, grid, 1, taps, Boundary::Reflect);
    check(out == std::vector<float>({1, 2, 1, 1, 2, 1}), "reflect edge");
    convolve_axis(input, out, grid, 1, taps, Boundary::Mirror);
    check(out == std::vector<float>({0, 2, 0, 0, 2, 0}), "mirror edge");
    convolve_axis(input, out, grid, 1, taps, Boundary::Zero);
    check(out == std::vector<float>({2, 2, -2, 5, 2, -5}), "zero contiguous edge");
    convolve_axis(input, out, grid, 0, taps, Boundary::Zero);
    check(out == std::vector<float>({4, 5, 6, -1, -2, -3}), "zero strided edge");
    convolve_axis(input, out, grid, 1, taps, Boundary::Nearest, true);
    check(out == std::vector<float>({-1, -2, -1, -1, -2, -1}), "reversed taps");
    const Grid line({1, 3});
    const std::vector<float> cancellation{16777216.f, 1.f, -16777216.f};
    out.resize(3);
    convolve_axis(cancellation, out, line, 1, std::vector<float>{1, 1, 1}, Boundary::Nearest);
    check(out[1] == 0, "image filters accumulate in image precision");
    filter_detail::correlate(cancellation.data(), out.data(), line, 1, {{1, 1, 1}, 1}, Boundary::Nearest);
    check(out[1] == 1, "derived filters retain double accumulation");
    filter_detail::correlate(cancellation.data(), out.data(), line, 0, {{.25, .5, .25}, 1}, Boundary::Mirror);
    check(out == cancellation, "singleton mirror axis");
}
void interpolation()
{
    const std::size_t shape[]{2, 2};
    const float data[]{0, 2, 4, 6};
    const double q[]{.25, .75};
    check(vcr::sampling::clamped<2, float, double>(data, shape, q) == 2.5f, "bilinear sample");
    check(vcr::sampling::clamped<2, float, double, float, true, true>(data, shape, q) == 2.5f, "reverse corner order");
    const double edge[]{-4, 10};
    check(vcr::sampling::clamped<2, float, double>(data, shape, edge) == 2, "edge replication");
    vcr::sampling::Cell<3> cell;
    const float point[]{-.25f, 1, 2.5f};
    vcr::sampling::cellOf(point, 3, cell, [](std::size_t) { return std::int64_t(1000000000); });
    check(cell.lo[0] == 999999999 && cell.frac[0] == .75f && cell.k == 2 && cell.axes[1] == 2,
          "local coordinates retain integer origin and omit zero-weight axes");
}
void percentiles()
{
    using namespace vcr::transforms::percentile_detail;
    std::vector<float> values{9, -1, 3, 7};
    check(percentile(values, 0) == -1 && percentile(values, 100) == 9, "percentile endpoints");
    check(percentile(values, 25) == 2 && percentile<Rule::Linear>(values, 75) == 7.5, "linear percentiles");
    check(sorted_percentile([&](std::size_t i) { return double(values[i]); }, 1, 50) == values[0], "singleton percentile");
}
void elements()
{
    using namespace vcr;
    using nlohmann::json;
    for (const auto* text : {"<u8", ">u8", "<i8", ">i8", "<f2", ">f2", "<f4", ">f4", "|b1"}) {
        const auto type = elementOf(text);
        const json value = type.kind == 'f' ? json(1.5) : type.kind == 'u' ? json(UINT64_MAX)
            : type.kind == 'i' ? json(INT64_MIN) : json(1);
        const auto bytes = encodeElement(value, type, ValueEncoding::Exact);
        check(decodeElement(bytes.data(), type) == value, "exact typed roundtrip including byte order");
    }
    rejects([] { (void)parseDtype("|b2"); });
    rejects([] { (void)parseDtype("<f1"); });
    rejects([] { (void)parseDtype("<u2garbage"); });
    rejects([] { (void)encodeElement(256, elementOf("|u1"), ValueEncoding::Exact); });
    check(encodeElement(256, elementOf("|u1"), ValueEncoding::ZarrFill)[0] == 0, "Zarr narrowing policy");
    const auto nan = encodeElement("NaN", elementOf("<f2"), ValueEncoding::ZarrFill);
    check(std::isnan(elementOf("<f2").load(nan.data())), "half NaN fill");
    const auto raw = encodeElement("0x3f800000", elementOf("<f4"), ValueEncoding::ZarrFill);
    check(elementOf("<f4").load(raw.data()) == 1, "v3 raw float fill");
}
void placement()
{
    using nlohmann::json;
    const json base = json::array({{{"type", "scale"}, {"scale", {1, 1, 1}}}});
    const json coarse = json::array({{{"type", "scale"}, {"scale", {2, 2, 2}}}, {{"type", "scale"}, {"scale", {3, 3, 3}}}});
    const auto inferred = vcr::sources::blockReducedTransforms(coarse, base);
    const auto resolved = vcr::sources::storageTransform({"z", "y", "x"}, {}, {inferred});
    const json attrs = {{"multiscales", json::array({{{"axes", {"z", "y", "x"}}, {"datasets", json::array({
        {{"path", "0"}, {"coordinateTransformations", base}}, {{"path", "1"}, {"coordinateTransformations", coarse}}
    })}}})}};
    const auto placed = vcr::zarr::levelPlacement(attrs, "1");
    for (int a = 0; a < 3; ++a) {
        check(placed.scale[a] == 6 && placed.offset[a] == 2.5, "composed pyramid scale and centre");
        check(resolved.transform->forward(a, a) == 6 && resolved.transform->forward(a, 3) == 2.5,
              "source registration and level placement agree");
    }
    const auto translated = vcr::sources::storageTransform({"y", "x"}, {}, {json::array({
        {{"type", "translation"}, {"translation", {2, 3}}}, {{"type", "scale"}, {"scale", {4, 5}}}
    })});
    check(translated.transform->forward(0, 3) == 15 && translated.transform->forward(1, 3) == 8, "transform order");
}
void atomicFiles()
{
    namespace fs = std::filesystem;
    const auto path = fs::temp_directory_path() / ("vcr-atomic-" + std::to_string(::getpid()));
    struct Cleanup { fs::path p; ~Cleanup() { std::error_code ec; fs::remove(p, ec); } } cleanup{path};
    const std::uint8_t first[]{1, 2, 3};
    vcr::atomic_write(path.string(), first, 3);
    std::string temporary;
    {
        vcr::AtomicFile abandoned(path.string()); temporary = abandoned.temporary_path();
        abandoned.write(first, 1);
    }
    check(!fs::exists(temporary) && fs::file_size(path) == 3, "aborted publication cleans up and preserves destination");
    vcr::atomic_write(path.string(), first, 1);
    check(fs::file_size(path) == 1, "atomic replacement");
    const fs::path relative = "vcr-atomic-relative-" + std::to_string(::getpid());
    Cleanup relativeCleanup{relative};
    vcr::atomic_write(relative.string(), first, 3);
    check(fs::file_size(relative) == 3, "publication to a bare relative filename");
}
}
int main()
{
    try {
        caches(); filters(); interpolation(); percentiles(); elements(); placement(); atomicFiles();
        std::cout << "Shared algorithm regression checks passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
