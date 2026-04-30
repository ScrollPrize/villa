#include "vc/core/util/VolpkgConvert.hpp"

#include <filesystem>
#include <iostream>
#include <string>

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "usage: " << argv[0] << " <input> <output.volpkg.json>\n"
                  << "  <input>: local volpkg directory or s3://, https:// URL\n";
        return 2;
    }

    auto r = vc::convertVolpkg(argv[1], std::filesystem::path(argv[2]));
    if (!r.ok) {
        std::cerr << r.message << "\n";
        return 1;
    }
    if (!r.message.empty()) std::cerr << "warning: " << r.message << "\n";
    std::cout << "wrote " << r.output << "\n";
    return 0;
}
