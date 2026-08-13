// Forwarding shim: <doctest/doctest.h> -> the in-tree minimal vc_test.hpp.
// Lets existing tests keep `#include <doctest/doctest.h>` unchanged.
#pragma once
#include <vc_test.hpp>
