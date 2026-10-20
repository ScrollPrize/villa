#pragma once
#include "options.hpp"

namespace vcr::transforms {
struct ForegroundCrop {Shape shape;std::string mask="mask";std::size_t channel=0;double threshold=0,p=1;};
} // namespace vcr::transforms
