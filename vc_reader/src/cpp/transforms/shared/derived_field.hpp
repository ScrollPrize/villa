#pragma once
#include "../core.hpp"

namespace vcr::transforms::derived_field_detail {
inline Target derived(Role role, const std::string& components = "zyx") { Target t; t.role = role; t.components = components; return t; }
inline Array metadata(const Shape& shape, std::size_t bytes) { Array a; a.shape = shape; a.kind = 'f'; a.bytes = bytes; return a; }
} // namespace vcr::transforms::derived_field_detail
