#include <nlohmann/json.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

int main(int argc, char** argv)
{
    std::unordered_map<std::string, std::vector<std::string>> options;
    for (int index = 1; index < argc;) {
        const std::string name = argv[index++];
        if (name == "--seed") {
            if (index + 2 >= argc)
                return 2;
            options[name] = {argv[index], argv[index + 1], argv[index + 2]};
            index += 3;
        } else {
            if (index >= argc)
                return 2;
            options[name] = {argv[index++]};
        }
    }
    for (const auto* required : {"--volume", "--target-dir", "--params", "--seed", "--segment-name"})
        if (!options.contains(required))
            return 2;

    const auto profile = nlohmann::json::parse(std::ifstream(options["--params"].front()));
    const int generations = profile.at("generations").get<int>();
    if (profile.at("use_cuda") != false || generations < 1)
        return 3;
    if (!std::filesystem::is_directory(options["--volume"].front()))
        return 4;

    if (generations >= 9999) {
        std::cout << "long-running fake growth" << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(10));
    }

    const std::filesystem::path output(options["--target-dir"].front());
    std::filesystem::create_directories(output);
    std::ofstream(output / "meta.json") << nlohmann::json({{"format", "tifxyz"}, {"name", options["--segment-name"].front()}, {"seed", options["--seed"]}})
                                               .dump(2)
                                        << '\n';
    std::cout << "generation 1" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    std::cout << "complete" << std::endl;
    return 0;
}
