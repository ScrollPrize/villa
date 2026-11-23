#pragma once
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace vc::objects {
// Schema-22 stable built-in codes. Do not reorder or reuse a code.
inline constexpr std::array<std::string_view,10> kCoreEdgeKinds{
    "parallel","perpendicular","same-surface","point-surface",
    "crossing","fiber_branch","attachment","declared_reference",
    "point_link","collection_winding"};
inline int64_t edgeKindCode(std::string_view name) {
    for(size_t i=0;i<kCoreEdgeKinds.size();++i)if(kCoreEdgeKinds[i]==name)return int64_t(i);
    throw std::invalid_argument("unsupported schema-22 edge kind: "+std::string(name));
}
inline std::string edgeKindName(int64_t code) {
    if(code<0||uint64_t(code)>=kCoreEdgeKinds.size())
        throw std::invalid_argument("unsupported schema-22 edge kind code: "+std::to_string(code));
    return std::string(kCoreEdgeKinds[size_t(code)]);
}
inline int64_t edgeApprovalCode(std::string_view name) {
    if(name.empty()||name=="pending")return 0;
    if(name=="approved")return 1;
    if(name=="rejected")return 2;
    throw std::invalid_argument("invalid edge approval state: "+std::string(name));
}
inline std::string edgeApprovalName(int64_t code) {
    if(code==0)return "pending";
    if(code==1)return "approved";
    if(code==2)return "rejected";
    throw std::invalid_argument("invalid edge approval code");
}
} // namespace vc::objects
