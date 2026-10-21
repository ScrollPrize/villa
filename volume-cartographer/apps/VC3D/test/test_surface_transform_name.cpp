#include "SurfaceTransformName.hpp"

#include <iostream>
#include <utility>
#include <vector>

int main()
{
    const QRegularExpression valid(QStringLiteral("^[a-zA-Z0-9_-]+$"));
    const std::vector<std::pair<QString, QString>> cases{
        {"spiral-output-generation-19-lasagna.tifxyz", "spiral-output-generation-19-lasagna_tifxyz_transformed"},
        {"segment_123", "segment_123_transformed"},
        {"  a b/c.d  ", "a_b_c_d_transformed"},
        {"", "surface_transformed"},
        {"   ", "surface_transformed"},
        {"../", "__transformed"},
    };
    for (const auto& [source, expected] : cases) {
        const auto name = suggestedTransformedSurfaceName(source);
        if (name != expected || !valid.match(name).hasMatch()) {
            std::cerr << "Invalid default for " << source.toStdString() << ": " << name.toStdString() << '\n';
            return 1;
        }
    }
}
