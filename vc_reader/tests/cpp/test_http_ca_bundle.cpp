#include "../../src/cpp/http.hpp"
#include <cstdlib>
#include <iostream>

// libcurl's own bundle is kept when it exists or when it has none; a missing one is
// replaced only by a file that exists.
int main(int, char** argv)
{
    // A file that exists wherever the test runs from: this executable (__FILE__ is
    // relative to the build tree in some builds).
    const auto self = std::filesystem::absolute(argv[0]).string();
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
