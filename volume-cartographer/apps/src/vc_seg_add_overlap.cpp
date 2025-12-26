#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include <nlohmann/json.hpp>
#include <fstream>


using json = nlohmann::json;



int main(int argc, char *argv[])
{
    if (argc != 3) {
        std::cout << "usage: " << argv[0] << " <tgt-dir> <single-tiffxyz>" << std::endl;
        std::cout << "   this will check for overlap between any tiffxyz in target dir and <single-tiffxyz> and add overlap metadata" << std::endl;
        return EXIT_SUCCESS;
    }

    std::filesystem::path tgt_dir = argv[1];
    std::filesystem::path seg_dir = argv[2];
    int search_iters = 10;
    srand(clock());

    QuadSurface current(seg_dir);

    // Read existing overlapping data for current segment
    std::set<std::string> current_overlapping = read_overlapping_json(current.path);

    bool found_overlaps = false;

    for (const auto& entry : std::filesystem::directory_iterator(tgt_dir))
        if (std::filesystem::is_directory(entry))
        {
            std::string name = entry.path().filename();
            if (name == current.id)
                continue;

            std::filesystem::path meta_fn = entry.path() / "meta.json";
            if (!std::filesystem::exists(meta_fn))
                continue;

            std::ifstream meta_f(meta_fn);
            json meta;
            try {
                meta = json::parse(meta_f);
            } catch (const json::exception& e) {
                std::cerr << "Error parsing meta.json for " << name << ": " << e.what() << std::endl;
                continue;
            }

            if (!meta.count("bbox"))
                continue;

            if (meta.value("format","NONE") != "tifxyz")
                continue;

            QuadSurface other = QuadSurface(entry.path(), meta);

            if (overlap(current, other, search_iters)) {
                found_overlaps = true;

                // Add to current's overlapping set
                current_overlapping.insert(other.id);

                // Read and update other's overlapping set
                std::set<std::string> other_overlapping = read_overlapping_json(other.path);
                other_overlapping.insert(current.id);
                write_overlapping_json(other.path, other_overlapping);

                std::cout << "Found overlap: " << current.id << " <-> " << other.id << std::endl;
            }
        }

    // Write current's overlapping data
    if (found_overlaps || !current_overlapping.empty()) {
        write_overlapping_json(current.path, current_overlapping);
        std::cout << "Updated overlapping data for " << current.id
                  << " (" << current_overlapping.size() << " overlaps)" << std::endl;
    }

    return EXIT_SUCCESS;
}