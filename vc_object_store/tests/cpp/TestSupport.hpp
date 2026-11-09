#pragma once
#include "vc_object_store/Uuid.hpp"
#include <filesystem>
#include <stdexcept>
#include <source_location>

inline void require(bool value, const char* message)
{
    if (!value) throw std::runtime_error(message);
}
template<class F> void mustFail(F&& callback, const std::source_location location=std::source_location::current())
{
    try { callback(); } catch (const std::exception&) { return; }
    throw std::runtime_error(std::string("expected operation to fail at ")+location.file_name()+":"+std::to_string(location.line()));
}
struct TemporaryDirectory {
    std::filesystem::path path = std::filesystem::temp_directory_path() / vc::objects::util::newUuid();
    TemporaryDirectory() { std::filesystem::create_directory(path); }
    ~TemporaryDirectory() { std::error_code error; std::filesystem::remove_all(path, error); }
};
