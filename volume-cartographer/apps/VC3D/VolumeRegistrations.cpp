#include "VolumeRegistrations.hpp"

#include <nlohmann/json.hpp>

#include <regex>
#include <stdexcept>

namespace vc3d::registrations {

vc::objects::VolumeAffine parseAffine(const std::string& text)
{
    static const std::regex number(R"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)");
    std::vector<double> values;
    for (auto it = std::sregex_iterator(text.begin(), text.end(), number); it != std::sregex_iterator(); ++it)
        values.push_back(std::stod(it->str()));
    if (values.size() != 12 && values.size() != 16)
        throw std::invalid_argument("expected three or four rows of four numbers, got " + std::to_string(values.size()) + " numbers");
    auto rows = nlohmann::json::array();
    for (size_t row = 0; row < values.size() / 4; ++row)
        rows.push_back({values[row * 4], values[row * 4 + 1], values[row * 4 + 2], values[row * 4 + 3]});
    // fromJson checks the homogeneous row and that the matrix inverts.
    return vc::objects::VolumeAffine::fromJson(rows);
}

} // namespace vc3d::registrations
