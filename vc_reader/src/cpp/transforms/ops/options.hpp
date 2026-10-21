#pragma once
#include "../core.hpp"

namespace vcr::transforms {
struct LocalOptions { Scalar scale = Uniform{10, 30}, loc = Uniform{-1, 2}; };
}

namespace vcr::transforms::volume_detail {
struct ReferenceVolume {Shape shape;Array values;bool wide=true;};
}

namespace vcr::transforms::volume_detail {
struct EraseValues {double image=0,label=0;bool labels=false;};
}
