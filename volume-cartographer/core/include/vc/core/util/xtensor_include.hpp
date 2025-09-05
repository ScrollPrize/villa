// xtensor_include.hpp
#pragma once

#if __has_include(<xtensor/xtensor_config.hpp>)
#include <xtensor/xtensor_config.hpp>
#elif __has_include(<xtensor/core/xtensor_config.hpp>)
#include <xtensor/core/xtensor_config.hpp>
#endif

#define _VC_XTENSOR_STR(x) #x
#define _VC_XTENSOR_JOIN_PATH(a, b) _VC_XTENSOR_STR(a/b)

#if defined(XTENSOR_VERSION_MAJOR) && defined(XTENSOR_VERSION_MINOR) && (XTENSOR_VERSION_MAJOR > 0 || XTENSOR_VERSION_MINOR >= 26)
  #define XTENSORINCLUDE(category, file) _VC_XTENSOR_JOIN_PATH(xtensor/category, file)
#else
  // Drop the category (e.g. "containers")
  #define XTENSORINCLUDE(category, file) _VC_XTENSOR_JOIN_PATH(xtensor, file)
#endif