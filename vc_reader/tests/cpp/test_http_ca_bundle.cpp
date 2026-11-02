#include "../../src/cpp/http.hpp"
#include <cstdlib>
#include <iostream>

// libcurl's own bundle is kept when it exists or when it has none; a missing one is
// replaced only by a file that exists.
int main()
{
    const auto self = std::filesystem::absolute(__FILE__).string();
    if (!vcr::select_ca_bundle("").empty() || !vcr::select_ca_bundle(self).empty()) {
        std::cerr << "kept bundle was replaced\n";
        return EXIT_FAILURE;
    }
    const auto replacement = vcr::select_ca_bundle("/nonexistent/ca-bundle.crt");
    if (!replacement.empty() && !std::filesystem::is_regular_file(replacement)) {
        std::cerr << "replacement bundle does not exist\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
