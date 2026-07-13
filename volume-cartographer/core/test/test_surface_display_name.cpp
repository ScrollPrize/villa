#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "SurfaceDisplayName.hpp"

TEST_CASE("Open-data surfaces use their original name as the panel label")
{
    const std::string internalId =
        "PHerc0139-20260311000000-published-volume-L2-0123456789abcdef";

    utils::Json catalogMetadata = {
        {"name", "seg-a"},
        {"vc_open_data_segment_id", "20260311000000"},
    };
    CHECK(vc3d::surfacePanelDisplayName(internalId, catalogMetadata) == "seg-a");

    catalogMetadata.erase("name");
    CHECK(vc3d::surfacePanelDisplayName(internalId, catalogMetadata) ==
          "20260311000000");

    const utils::Json localMetadata = {{"name", "local-friendly-name"}};
    CHECK(vc3d::surfacePanelDisplayName(internalId, localMetadata) == internalId);
}
