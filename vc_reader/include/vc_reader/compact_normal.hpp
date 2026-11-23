#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>

namespace vcr::normal {
// Lasagna's two-channel axis representation chooses the positive-Z hemisphere.
// Double arithmetic preserves the established quantizer's byte boundaries.
inline std::optional<std::array<uint8_t,2>> encodeCompactAxis(const std::array<double,3>& input) {
    const double length=std::sqrt(input[0]*input[0]+input[1]*input[1]+input[2]*input[2]);
    if(!(length>1e-12)||!std::isfinite(length))return std::nullopt;
    std::array<double,3> normal{input[0]/length,input[1]/length,input[2]/length};
    if(normal[2]<0.)for(auto& value:normal)value=-value;
    const auto encode=[](double v){return uint8_t(std::clamp(std::lround(v*127.+128.),0L,255L));};
    return std::array<uint8_t,2>{encode(normal[0]),encode(normal[1])};
}
inline std::array<double,3> decodeCompactAxis(double x,double y) {
    const double nx=(x-128.)/127.,ny=(y-128.)/127.;
    const double nz=std::sqrt(std::max(0.,1.-nx*nx-ny*ny));
    const double length=std::sqrt(nx*nx+ny*ny+nz*nz);
    if(!(length>1e-12)||!std::isfinite(length))return {};
    return {nx/length,ny/length,nz/length};
}
inline std::array<uint8_t,3> encodeSignedCompact(const std::array<double,3>& normal) {
    const auto axis=encodeCompactAxis(normal);
    if(!axis)return {};
    return {(*axis)[0],(*axis)[1],uint8_t(normal[2]<0.?255:1)};
}
inline std::array<double,3> decodeSignedCompact(const std::array<uint8_t,3>& bytes) {
    if(bytes[2]==0)return {};
    if(bytes[2]!=1&&bytes[2]!=255)throw std::invalid_argument("invalid signed compact-normal hemisphere");
    auto normal=decodeCompactAxis(bytes[0],bytes[1]);
    if(bytes[2]==255)for(auto& value:normal)value=-value;
    return normal;
}
} // namespace vcr::normal
