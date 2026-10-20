#include "../cpu/labels.hpp"
#include "resident.cuh"
namespace vcr::transforms::cuda {
void apply_cuda(const ToPseudo2D& t, Sample& s, const Parameters& p, Session& session) {
    // Shared metadata-only reshape: no voxel reads or transfers.
    apply(t,s,p);
    for (const auto& c : p.channels) session.arrays.at(c.key).shape = s.at(c.key).array.shape;
}
}
