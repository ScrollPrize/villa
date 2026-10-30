#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <stdexcept>
#include <system_error>

int vc_normal_cache_fixture_main(int argc, char** argv);

int main()
{
    namespace fs = std::filesystem;
    fs::path owned;
    int result = 2;
    try {
        std::random_device random;
        for (int attempt = 0; attempt < 32 && owned.empty(); ++attempt) {
            const auto candidate = fs::temp_directory_path() /
                ("vc-normal-cache-" + std::to_string(random()) + "-" +
                 std::to_string(random()));
            if (fs::create_directory(candidate)) owned = candidate;
        }
        if (owned.empty()) throw std::runtime_error("cannot create exclusive test directory");
        std::string grids = (owned / "grids").string();
        char program[] = "test_normal_constraint_grid_change";
        char* arguments[] = {program, grids.data(), nullptr};
        result = vc_normal_cache_fixture_main(2, arguments);
    } catch (const std::exception& error) {
        std::cerr << "test setup error: " << error.what() << '\n';
    }
    if (!owned.empty()) {
        std::error_code error;
        fs::remove_all(owned, error);
        if (error) {
            std::cerr << "test cleanup error: " << error.message() << '\n';
            return 2;
        }
    }
    return result;
}
