#define VC_TEST_MAIN
#include "vc_test.hpp"
#include "vc_object_store/TerminalProgress.hpp"
#include <condition_variable>
#include <sstream>

using namespace vc::objects::util;

TEST_CASE("terminal progress reports final counts on stderr-style streams")
{
    std::ostringstream output;
    TerminalProgress progress("Test", true, output, false, std::chrono::hours(1));
    auto callback = progress.callback();
    reportProgress(callback, "loading", 0, 2);
    reportProgress(callback, "loading", 1, 2); // throttled
    reportProgress(callback, "loading", 2, 2); // finish flushes the latest state
    progress.finish();
    const auto text = output.str();
    CHECK(text.find("2/2") != std::string::npos);
    CHECK(text.find("100.0%") != std::string::npos);
    CHECK(text.find("loading") != std::string::npos);
    CHECK(text.find("elapsed") != std::string::npos);
    CHECK(text.find("(finished)") != std::string::npos);
    CHECK(text.find('\r') == std::string::npos);
    CHECK(text.find('\033') == std::string::npos);
    reportProgress(callback, "ignored after finish", 99, 100);
    CHECK(output.str() == text);
}

TEST_CASE("terminal progress keeps reporting during work without callbacks")
{
    struct Buffer : std::stringbuf {
        std::mutex mutex;
        std::condition_variable changed;
        int flushes = 0;
        int sync() override {
            { std::lock_guard lock(mutex); ++flushes; }
            changed.notify_all();
            return 0;
        }
    } buffer;
    std::ostream output(&buffer);
    TerminalProgress progress("Slow index", true, output, false, std::chrono::milliseconds(10));
    {
        std::unique_lock lock(buffer.mutex);
        CHECK(buffer.changed.wait_for(lock, std::chrono::seconds(2), [&] { return buffer.flushes >= 2; }));
    }
    progress.finish(); // joins the renderer before reading its buffer
    CHECK(buffer.str().find("100.0%") == std::string::npos);
    CHECK(buffer.str().find("elapsed") != std::string::npos);
}

TEST_CASE("terminal progress handles quiet mode and interrupted operations")
{
    std::ostringstream output;
    {
        TerminalProgress quiet("Quiet", false, output);
        CHECK_FALSE(bool(quiet.callback()));
        quiet.update({"hidden", 1, 2});
    }
    CHECK(output.str().empty());
    {
        TerminalProgress progress("Test", true, output, true, std::chrono::hours(1));
        progress.update({"loading", 1, 2});
    }
    CHECK(output.str().find('\r') != std::string::npos);
    CHECK(output.str().find("interrupted") != std::string::npos);
    CHECK(output.str().back() == '\n');
    CHECK(progressBar(100, 1) == "[########################]");
    CHECK(progressBar(0, 0) == "[------------------------]");
}
