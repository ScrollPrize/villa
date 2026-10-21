#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "../src/render/SamplingWaitProfile.hpp"

#include <future>
#include <iostream>
#include <sstream>
#include <vector>

TEST_CASE("Sampling diagnostics are optional and serialize thread-local summaries")
{
    using namespace vc::render::detail;
    struct Capture {
        std::ostringstream text;
        std::streambuf* previous = std::cerr.rdbuf(text.rdbuf());
        ~Capture() { std::cerr.rdbuf(previous); }
    } capture;
    const bool enabled = samplingWaitProfileEnabled();
    auto& mainStats = samplingWaitStats();
    mainStats = {};
    mainStats.last -= std::chrono::seconds(11);
    const auto oldLast = mainStats.last;

    recordSamplingWait(0, 0.25);
    CHECK(mainStats.calls[0] == (enabled ? 1 : 0));
    CHECK(mainStats.seconds[0] == (enabled ? 0.25 : 0.0));
    flushSamplingWaits();
    CHECK(mainStats.calls[0] == 0);
    CHECK(mainStats.seconds[0] == 0.0);
    CHECK((mainStats.last > oldLast) == enabled);
    const auto firstOutput = capture.text.str();
    flushSamplingWaits(); // Must not emit again before the interval expires.
    CHECK(capture.text.str() == firstOutput);

    std::vector<std::future<bool>> workers;
    for (int i = 0; i < 8; ++i) {
        workers.push_back(std::async(std::launch::async, [] {
            auto& stats = samplingWaitStats();
            const bool fresh = stats.calls[0] == 0 && stats.seconds[0] == 0;
            stats.last -= std::chrono::seconds(11);
            recordSamplingWait(0, 0.25);
            flushSamplingWaits();
            return fresh && stats.calls[0] == 0 && stats.seconds[0] == 0;
        }));
    }
    for (auto& worker : workers)
        CHECK(worker.get());
    CHECK(mainStats.calls[0] == 0);

    if (!enabled) {
        CHECK(capture.text.str().empty());
    } else {
        std::istringstream lines(capture.text.str());
        std::string line;
        int count = 0;
        while (std::getline(lines, line)) {
            ++count;
            CHECK(line.find("sampling-waits thread=") == 0);
            CHECK(line.find(" blocking_total=0.25/1 ") != std::string::npos);
            CHECK(line.ends_with(" shared_budget=0/0"));
            CHECK(line.find("sampling-waits", 1) == std::string::npos);
        }
        CHECK(count == 9);
    }
}
