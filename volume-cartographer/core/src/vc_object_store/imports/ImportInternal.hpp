#pragma once
#include "vc/core/objects/ObjectStore.hpp"
#include <fstream>
#include <sstream>

namespace vc::objects { namespace {
std::string readFile(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot read file: " + path.string());
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string relativeToDb(ObjectStore& db, const std::filesystem::path& file)
{
    std::error_code ec;
    auto rel = std::filesystem::relative(file, db.path().parent_path(), ec);
    if (ec) return file.filename().string();
    return rel.string();
}

} }
