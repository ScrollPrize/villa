#include "vc/core/util/Logging.hpp"

#include <memory>

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/dist_sink.h>
#include <spdlog/sinks/stdout_sinks.h>


auto DistSink() -> std::shared_ptr<spdlog::sinks::dist_sink_mt>;
auto DistSink() -> std::shared_ptr<spdlog::sinks::dist_sink_mt>
{
    static auto loggers = std::make_shared<spdlog::sinks::dist_sink_mt>();
    return loggers;
}

static auto Init() -> std::shared_ptr<spdlog::sinks::dist_sink_mt>
{
    DistSink()->add_sink(std::make_shared<spdlog::sinks::stdout_sink_mt>());
    return DistSink();
}


auto Logger() -> std::shared_ptr<spdlog::logger>
{
    static auto logger = std::make_shared<spdlog::logger>("volcart", Init());
    return logger;
}