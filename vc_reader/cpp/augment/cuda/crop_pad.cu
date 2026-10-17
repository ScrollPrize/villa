#include "geometry.cuh"
namespace vcr::augment::cuda {
void apply_cuda(const CropPad&, Sample& s, const Parameters& p, Session& session) { geometry_cuda(s,p,session); }
}
