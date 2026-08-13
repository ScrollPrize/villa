// Coverage for inline helpers in core/include/vc/core/util/Zarr.hpp.
// mapTileIndex is a pure integer transform; easy to assert exhaustively
// over the 4-rotation × 4-flip option space.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/Zarr.hpp"

TEST_CASE("mapTileIndex: no rotation, no flip is identity")
{
    int outTx, outTy, outTilesX, outTilesY;
    mapTileIndex(3, 5, /*tilesX=*/10, /*tilesY=*/20,
                 /*quadRot=*/0, /*flipType=*/-1,
                 outTx, outTy, outTilesX, outTilesY);
    CHECK(outTx == 3);
    CHECK(outTy == 5);
    CHECK(outTilesX == 10);
    CHECK(outTilesY == 20);
}

TEST_CASE("mapTileIndex: rotate 180 about centre, no flip")
{
    int outTx, outTy, outTilesX, outTilesY;
    // tiles 10x20, rot=2 (180°): tx -> 9-tx, ty -> 19-ty
    mapTileIndex(3, 5, 10, 20, /*quadRot=*/2, /*flipType=*/-1,
                 outTx, outTy, outTilesX, outTilesY);
    CHECK(outTx == 6);
    CHECK(outTy == 14);
    CHECK(outTilesX == 10);
    CHECK(outTilesY == 20);
}

TEST_CASE("mapTileIndex: rotate 90° swaps the tilesX/Y output")
{
    int outTx, outTy, outTilesX, outTilesY;
    mapTileIndex(3, 5, 10, 20, /*quadRot=*/1, -1,
                 outTx, outTy, outTilesX, outTilesY);
    // case 1: rx = ty, ry = tilesX-1-tx
    CHECK(outTx == 5);
    CHECK(outTy == 10 - 1 - 3); // 6
    // tilesX/Y swapped
    CHECK(outTilesX == 20);
    CHECK(outTilesY == 10);
}

TEST_CASE("mapTileIndex: rotate 270° swaps and inverts")
{
    int outTx, outTy, outTilesX, outTilesY;
    mapTileIndex(3, 5, 10, 20, /*quadRot=*/3, -1,
                 outTx, outTy, outTilesX, outTilesY);
    // case 3: rx = tilesY-1-ty, ry = tx
    CHECK(outTx == 20 - 1 - 5); // 14
    CHECK(outTy == 3);
    CHECK(outTilesX == 20);
    CHECK(outTilesY == 10);
}

TEST_CASE("mapTileIndex: flipType=0 mirrors Y; flipType=1 mirrors X; flipType=2 both")
{
    int outTx, outTy, outTilesX, outTilesY;
    // flip 0: fy = rTY-1-ry
    mapTileIndex(3, 5, 10, 20, 0, 0, outTx, outTy, outTilesX, outTilesY);
    CHECK(outTx == 3);
    CHECK(outTy == 20 - 1 - 5); // 14

    // flip 1: fx = rTX-1-rx
    mapTileIndex(3, 5, 10, 20, 0, 1, outTx, outTy, outTilesX, outTilesY);
    CHECK(outTx == 10 - 1 - 3); // 6
    CHECK(outTy == 5);

    // flip 2: both
    mapTileIndex(3, 5, 10, 20, 0, 2, outTx, outTy, outTilesX, outTilesY);
    CHECK(outTx == 6);
    CHECK(outTy == 14);
}

TEST_CASE("mapTileIndex: rot+flip composition is consistent")
{
    int a_tx, a_ty, a_tX, a_tY;
    int b_tx, b_ty, b_tX, b_tY;
    // 180° rotation + Y-flip should equal X-flip alone (mod tilesX/Y).
    mapTileIndex(3, 5, 10, 20, 2, 0, a_tx, a_ty, a_tX, a_tY);
    mapTileIndex(3, 5, 10, 20, 0, 1, b_tx, b_ty, b_tX, b_tY);
    CHECK(a_tx == b_tx);
    CHECK(a_ty == b_ty);
}
