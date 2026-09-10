#include "../../apps/src/SamplingGridCapture.hpp"
#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>

using Capture = vc::render::SamplingGridCapture;
static void require(bool ok) { if (!ok) throw std::runtime_error("test assertion failed"); }
template<class F> static void rejects(F f) {
    bool failed = false;
    try { f(); } catch (const std::exception&) { failed = true; }
    require(failed);
}
static Capture::Pixel pixel(int r, int c) {
    return {float(c), float(r), 1.25f, 0.0f, 0.0f, 1.0f};
}
static std::string serialized(const Capture& capture) {
    std::ostringstream out;
    capture.writeJson(out);
    return out.str();
}
static void selftest() {
    rejects([] { Capture c(1, 2, 0, 0, 0); });
    rejects([] { Capture c(513, 513, 0, 0, 0); });
    rejects([] { Capture c(2, 2, -1, 0, 0); });
    Capture all(4, 3, 2, 7, 8), bands(4, 3, 2, 7, 8);
    const std::vector<float> offsets{-0.5f, -0.25f, 0.5f, 0.75f};
    all.appendBand(0, 4, 3, offsets, pixel);
    // Out-of-order bands, including a seam: no implicit row-order assumption.
    bands.appendBand(2, 2, 3, offsets, [](int r, int c) { return pixel(r+2, c); });
    rejects([&] { serialized(bands); });
    bands.appendBand(0, 2, 3, offsets, pixel);
    require(serialized(all) == serialized(bands));
    rejects([&] { bands.appendBand(0, 1, 3, offsets, pixel); });
    Capture bad(2, 2, 0, 0, 0);
    rejects([&] { bad.appendBand(-1, 1, 2, offsets, pixel); });
    rejects([&] { bad.appendBand(0, 3, 2, offsets, pixel); });
    rejects([&] { bad.appendBand(0, 1, 3, offsets, pixel); });
    rejects([&] { bad.appendBand(0, 1, 2, {}, pixel); });
    rejects([&] { bad.appendBand(0, 1, 2, {NAN}, pixel); });
    bad.appendBand(0, 1, 2, offsets, pixel);
    rejects([&] { bad.appendBand(1, 1, 2, {0.0f}, pixel); });
    bad.appendBand(1, 1, 2, offsets, [](int, int) {
        return Capture::Pixel{NAN, 2, 3, 0, 0, 1};
    });
    auto invalid = serialized(bad);
    require(invalid.find("null") != std::string::npos);
    require(invalid.find("\"invalid_pixels\":2") != std::string::npos);
    require(invalid.find("[-0.5,-0.25,0.5,0.75]") != std::string::npos);
    require(invalid.find("\"shape_hw\":[2,2]") != std::string::npos);
    Capture signedZero(2, 2, 0, 0, 0);
    signedZero.appendBand(0, 2, 2, {-0.0f}, [](int, int) {
        return Capture::Pixel{-0.0f, 0, 1, 0, 0, 1};
    });
    require(serialized(signedZero).find("\"offsets\":[-0.0]") != std::string::npos);
    require(serialized(signedZero).find("[-0.0,0,1,0,0,1]") != std::string::npos);
    // A failed stream may not look like a completed capture.
    std::ostringstream broken; broken.setstate(std::ios::badbit);
    rejects([&] { all.writeJson(broken); });
    std::cout << "Capture selftests passed: dimensions, coverage, band seams/order, duplicate rows, bounds, offsets, invalid preservation, stream failure\n";
}
int main(int argc, char** argv) {
    try {
        if (argc == 1 || (argc == 2 && std::string(argv[1]) == "selftest")) { selftest(); return 0; }
        if (argc != 2 || std::string(argv[1]) != "replay")
            throw std::invalid_argument("use selftest or replay; replay consumes a research fixture, NOT a renderer execution");
        int h, w, level, x, y, count;
        if (!(std::cin >> h >> w >> level >> x >> y >> count) || count < 1 || count > 65536)
            throw std::invalid_argument("invalid replay header");
        Capture capture(h, w, level, x, y);
        std::vector<float> offsets(std::size_t(count), 0.0f);
        for (float& f : offsets) if (!(std::cin >> f)) throw std::runtime_error("truncated offsets");
        std::vector<Capture::Pixel> pixels(std::size_t(h) * std::size_t(w));
        for (auto& p : pixels)
            for (float& f : p)
                if (!(std::cin >> f)) throw std::runtime_error("truncated rays");
        std::string extra;
        if (std::cin >> extra) throw std::runtime_error("unexpected replay data");
        for (int y0 = 0; y0 < h; y0 += 37)
            capture.appendBand(y0, std::min(37, h-y0), w, offsets,
                [&](int r, int c) { return pixels[std::size_t(y0+r)*std::size_t(w)+std::size_t(c)]; });
        capture.writeJson(std::cout);
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
