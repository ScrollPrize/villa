#include "vc/core/types/ChunkedTensor.hpp"

#include "utils/Json.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>

std::filesystem::path read_cache_meta_dataset_path(const std::filesystem::path& meta_json_path)
{
    try {
        auto meta = utils::Json::parse_file(meta_json_path);
        if (!meta.contains("dataset_source_path") || !meta["dataset_source_path"].is_string())
            return {};
        std::filesystem::path src(meta["dataset_source_path"].get_string());
        if (!std::filesystem::exists(src))
            return {};
        return std::filesystem::canonical(src);
    } catch (const std::exception&) {
        return {};
    }
}

void write_cache_meta_json(const std::filesystem::path& dir, const std::filesystem::path& dataset_path)
{
    utils::Json meta = utils::Json::object();
    meta["dataset_source_path"] = dataset_path.string();
    std::ofstream o(dir / "meta.json");
    o << meta.dump(4) << std::endl;
}

void print_accessor_stats()
{
    std::cout << "acc miss/total " << miss << " " << total << " " << double(miss)/total << std::endl;
    std::cout << "chunk compute overhead/total " << chunk_compute_collisions << " " << chunk_compute_total << " " << double(chunk_compute_collisions)/chunk_compute_total << std::endl;
}
