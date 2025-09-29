#pragma once

// Shared enumeration describing how segmentation edit handles propagate their influence.
enum class SegmentationInfluenceMode
{
    GridChebyshev = 0,
    GeodesicCircular = 1,
    RowColumn = 2
};

// User-selectable behaviour when Row/Col influence mode is active.
enum class SegmentationRowColMode
{
    RowOnly = 0,
    ColumnOnly = 1,
    Dynamic = 2
};

// Per-handle axis preference used when Row/Col mode is active.
enum class SegmentationRowColAxis
{
    Both = 0,
    Row = 1,
    Column = 2
};

// Visibility behaviour for slice viewers when rendering segmentation handles.
enum class SegmentationSliceDisplayMode
{
    Fade = 0,
    Hide = 1
};
