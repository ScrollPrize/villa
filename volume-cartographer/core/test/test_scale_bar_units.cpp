#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "volume_viewers/CVolumeViewerView.hpp"

namespace {

std::string labelFor(double micrometers)
{
    return CVolumeViewerView::formatScaleBarLength(micrometers).text.toStdString();
}

} // namespace

TEST_CASE("scale bar formats micrometers with correct metric thresholds")
{
    CHECK(labelFor(500.0) == "500 µm");
    CHECK(labelFor(1000.0) == "1 mm");
    CHECK(labelFor(5000.0) == "5 mm");
    CHECK(labelFor(10000.0) == "1 cm");
    CHECK(labelFor(1000000.0) == "1 m");
}
