#pragma once

#include "vc/atlas/FiberHvClassification.hpp"

namespace vc3d::line_annotation {
using FiberHvTag = vc::atlas::FiberHvTag;
using FiberHvClassification = vc::atlas::FiberHvClassification;
using vc::atlas::classifyFiberHv;
using vc::atlas::fiberHvTagFromString;
using vc::atlas::fiberHvTagToString;
using vc::atlas::fiberLineLengthVx;
using vc::atlas::firstFiberDisplaysAsH;
} // namespace vc3d::line_annotation
