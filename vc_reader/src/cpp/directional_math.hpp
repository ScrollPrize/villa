#pragma once
#include <cmath>
#ifdef __CUDACC__
#define VCR_DIRECTION_HD __host__ __device__
#else
#define VCR_DIRECTION_HD
#endif
namespace vcr::direction {
template<class Real> VCR_DIRECTION_HD void apply(const Real* matrix, Real* value, int dim) {
    Real out[3]{};
    for (int i=0;i<dim;++i) for (int j=0;j<dim;++j) out[i]+=matrix[i*dim+j]*value[j];
    for (int i=0;i<dim;++i) value[i]=out[i];
}
template<class Real> VCR_DIRECTION_HD Real unit(Real* value, int dim) {
    Real norm=0;
    for (int i=0;i<dim;++i) norm+=value[i]*value[i];
    if (norm>0) for (int i=0;i<dim;++i) value[i]/=sqrt(norm);
    return norm;
}
}
#undef VCR_DIRECTION_HD
