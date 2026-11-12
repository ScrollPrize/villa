#pragma once

#include <nlohmann/json.hpp>
#include <filesystem>
#include <stdexcept>

// Shared command validation; axial-only feature generation is not a joint
// normal-extraction run and must not require or write normal-prior outputs.
inline bool validateFiberConstraintJointNormals(const nlohmann::json& cfg)
{
    const bool joint=cfg.value("orient_normals",false);
    if(joint && (cfg.value("repetitions",3)!=1 || !cfg.contains("database") || cfg.contains("orient_database") ||
       !cfg.contains("normal_prior_output") || !cfg.contains("umbilicus") || !cfg.contains("umbilicus_sha256")))
        throw std::invalid_argument("Joint normal extraction requires one database, umbilicus identity and prior output");
    if(joint && std::filesystem::exists(cfg.at("normal_prior_output").get<std::string>()))
        throw std::runtime_error("Normal prior output already exists");
    return joint;
}
