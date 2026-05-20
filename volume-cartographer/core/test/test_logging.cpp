#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/Logging.hpp"

#include <filesystem>
#include <fstream>
#include <random>
#include <string>

namespace fs = std::filesystem;

namespace {

fs::path tmpFile(const std::string& tag)
{
    std::mt19937_64 rng(std::random_device{}());
    return fs::temp_directory_path() /
           ("vc_log_" + tag + "_" + std::to_string(rng()) + ".log");
}

std::string readFile(const fs::path& p)
{
    std::ifstream in(p);
    return std::string((std::istreambuf_iterator<char>(in)),
                       std::istreambuf_iterator<char>());
}

} // namespace

TEST_CASE("Logger() returns a non-null shared instance")
{
    auto a = Logger();
    auto b = Logger();
    CHECK(a);
    CHECK(a.get() == b.get());
}

TEST_CASE("Logger string-overloads do not crash at default Info level")
{
    auto log = Logger();
    log->set_level(LogLevel::Info);
    log->debug(std::string("debug-msg"));  // filtered out
    log->info(std::string("info-msg"));
    log->warn(std::string("warn-msg"));
    log->error(std::string("error-msg"));
    CHECK(true);
}

TEST_CASE("Logger format-string overloads do not crash")
{
    auto log = Logger();
    log->set_level(LogLevel::Debug);
    log->debug("debug fmt {}", 1);
    log->info("info fmt {} {}", 2, "x");
    log->warn("warn fmt {}", 3.14);
    log->error("error fmt {}", "y");
    CHECK(true);
}

TEST_CASE("Logger::set_level filters lower-priority messages")
{
    auto log = Logger();
    auto path = tmpFile("filter");
    log->add_file(path);
    log->set_level(LogLevel::Warn);
    log->info("INFO_SHOULD_NOT_APPEAR_X");
    log->warn("WARN_SHOULD_APPEAR_X");
    auto content = readFile(path);
    CHECK(content.find("WARN_SHOULD_APPEAR_X") != std::string::npos);
    CHECK(content.find("INFO_SHOULD_NOT_APPEAR_X") == std::string::npos);
    // Reset so other tests aren't affected.
    log->set_level(LogLevel::Info);
    fs::remove(path);
}

TEST_CASE("add_file writes to disk")
{
    auto path = tmpFile("addfile");
    AddLogFile(path);
    auto log = Logger();
    log->set_level(LogLevel::Info);
    log->info("INFO_TO_FILE_ZQ");
    CHECK(fs::exists(path));
    auto content = readFile(path);
    CHECK(content.find("INFO_TO_FILE_ZQ") != std::string::npos);
    fs::remove(path);
}

TEST_CASE("add_file with bad path is a no-op (no throw)")
{
    auto log = Logger();
    log->add_file("/__nonexistent__/__path__/file.log");
    log->info("still works");
    CHECK(true);
}

TEST_CASE("SetLogLevel accepts all documented spellings")
{
    auto log = Logger();
    for (const char* s : {"debug", "DEBUG", "info", "INFO",
                          "warn", "WARN", "warning", "WARNING",
                          "error", "ERROR", "off", "OFF",
                          "unknown_garbage"}) {
        SetLogLevel(s);
    }
    // Restore default so other tests aren't affected.
    SetLogLevel("info");
    CHECK(true);
}

TEST_CASE("DebugLoggingEnabled flag round-trips")
{
    bool initial = DebugLoggingEnabled();
    SetDebugLoggingEnabled(true);
    CHECK(DebugLoggingEnabled());
    SetDebugLoggingEnabled(false);
    CHECK_FALSE(DebugLoggingEnabled());
    SetDebugLoggingEnabled(initial);
}

TEST_CASE("ProfileLoggingEnabled flag round-trips")
{
    bool initial = ProfileLoggingEnabled();
    SetProfileLoggingEnabled(true);
    CHECK(ProfileLoggingEnabled());
    SetProfileLoggingEnabled(false);
    CHECK_FALSE(ProfileLoggingEnabled());
    SetProfileLoggingEnabled(initial);
}
