#include "geometry.cuh"
namespace vcr::transforms::cuda {
void apply_cuda(const ThickSlice& t, Sample& s, const Parameters& p, Session& session) {
    for (const auto& [key,f] : s) if (f.target.role == Role::Image || (t.labels && (f.target.role == Role::Label || f.target.role == Role::Scalar))) {
        auto& a = session.arrays.at(key); auto shape = a.spatial();
        a = resize_cuda(resize_cuda(a,p.shape,true,true),shape,true,true);
    }
    session.restore_valid(s);
}
} // namespace vcr::transforms::cuda
