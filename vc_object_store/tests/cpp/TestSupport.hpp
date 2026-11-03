#pragma once
#include "vc_object_store/Uuid.hpp"
#include <filesystem>
#include <stdexcept>

inline void require(bool value, const char* message)
{
    if (!value) throw std::runtime_error(message);
}
template<class F> void mustFail(F&& callback)
{
    try { callback(); } catch (const std::exception&) { return; }
    throw std::runtime_error("expected operation to fail");
}
struct TemporaryDirectory {
    std::filesystem::path path = std::filesystem::temp_directory_path() / vc::objects::util::newUuid();
    TemporaryDirectory() { std::filesystem::create_directory(path); }
    ~TemporaryDirectory() { std::error_code error; std::filesystem::remove_all(path, error); }
};
