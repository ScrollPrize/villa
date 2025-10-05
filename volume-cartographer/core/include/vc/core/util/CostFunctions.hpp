#pragma once

#include "vc/core/types/ChunkedTensor.hpp"
#include "vc/core/util/NormalGridVolume.hpp"
#include "vc/core/util/GridStore.hpp"

#include <opencv2/core.hpp>

#include "ceres/ceres.h"


static double  val(const double &v) { return v; }
template <typename JetT>
double  val(const JetT &v) { return v.a; }

struct DistLoss {
    DistLoss(float dist, float w) : _d(dist), _w(w) {};
    template <typename T>
    bool operator()(const T* const a, const T* const b, T* residual) const {
        if (val(a[0]) == -1 && val(a[1]) == -1 && val(a[2]) == -1) {
            residual[0] = T(0);
            std::cout << "invalid DistLoss CORNER" << std::endl;
            return true;
        }
        if (val(b[0]) == -1 && val(b[1]) == -1 && val(b[2]) == -1) {
            residual[0] = T(0);
            std::cout << "invalid DistLoss CORNER" << std::endl;
            return true;
        }

        T d[3];
        d[0] = a[0] - b[0];
        d[1] = a[1] - b[1];
        d[2] = a[2] - b[2];

        T dist = sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);

        // Guard: near-zero-length segment leads to unstable division and duplicate-placement artifacts
        if (dist <= T(1e-12)) {
            std::cout << "[DistLoss:zero_distance] d=" << val(_d)
                      << " a=(" << val(a[0]) << "," << val(a[1]) << "," << val(a[2]) << ")"
                      << " b=(" << val(b[0]) << "," << val(b[1]) << "," << val(b[2]) << ")"
                      << std::endl;
            residual[0] = T(0);
            return true;
        }

        if (dist <= T(0)) {
            residual[0] = T(_w)*(d[0]*d[0] + d[1]*d[1] + d[2]*d[2] - T(1));
        }
        else {
            if (dist < T(_d))
                residual[0] = T(_w)*(T(_d)/dist - T(1));
            else
                residual[0] = T(_w)*(dist/T(_d) - T(1));
        }

        return true;
    }

    double _d;
    double _w;

    static ceres::CostFunction* Create(float d, float w = 1.0)
    {
        return new ceres::AutoDiffCostFunction<DistLoss, 1, 3, 3>(new DistLoss(d, w));
    }
};

struct DistLoss2D {
    DistLoss2D(float dist, float w) : _d(dist), _w(w) {};
    template <typename T>
    bool operator()(const T* const a, const T* const b, T* residual) const {
        if (val(a[0]) == -1 && val(a[1]) == -1 && val(a[2]) == -1) {
            residual[0] = T(0);
            std::cout << "invalid DistLoss2D CORNER" << std::endl;
            return true;
        }
        if (val(b[0]) == -1 && val(b[1]) == -1 && val(b[2]) == -1) {
            residual[0] = T(0);
            std::cout << "invalid DistLoss2D CORNER" << std::endl;
            return true;
        }

        T d[2];
        d[0] = a[0] - b[0];
        d[1] = a[1] - b[1];

        T dist = sqrt(d[0]*d[0] + d[1]*d[1]);

        // Guard: near-zero-length segment leads to unstable division and duplicate-placement artifacts
        if (dist <= T(1e-12)) {
            std::cout << "[DistLoss2D:zero_distance] d=" << val(_d)
                      << " a=(" << val(a[0]) << "," << val(a[1]) << ")"
                      << " b=(" << val(b[0]) << "," << val(b[1]) << ")"
                      << std::endl;
            residual[0] = T(0);
            return true;
        }

        if (dist <= T(0)) {
            residual[0] = T(_w)*(d[0]*d[0] + d[1]*d[1] - T(1));
            std::cout << "uhohh" << std::endl;
        }
        else {
            if (dist < T(_d))
                residual[0] = T(_w)*(T(_d)/(dist+T(1e-2)) - T(1));
            else
                residual[0] = T(_w)*(dist/T(_d) - T(1));
        }

        return true;
    }

    double _d;
    double _w;

    static ceres::CostFunction* Create(float d, float w = 1.0)
    {
        if (d == 0)
            throw std::runtime_error("dist can't be zero for DistLoss2D");
        return new ceres::AutoDiffCostFunction<DistLoss2D, 1, 2, 2>(new DistLoss2D(d, w));
    }
};



// Encourages movement of p relative to b along a precomputed outward direction.
struct ForwardProgressLoss {
    ForwardProgressLoss(const cv::Vec3d& outward, float w) : _w(w) {
        double n = std::sqrt(outward[0]*outward[0] + outward[1]*outward[1] + outward[2]*outward[2]);
        if (n > 1e-12) {
            _out[0] = outward[0]/n; _out[1] = outward[1]/n; _out[2] = outward[2]/n;
        } else {
            _out[0] = _out[1] = _out[2] = 0.0;
        }
    }
    template <typename T>
    bool operator()(const T* const p, const T* const b, T* residual) const {
        // Penalize negative projection onto outward direction; zero when moving outward or perpendicular
        T move0 = p[0] - b[0];
        T move1 = p[1] - b[1];
        T move2 = p[2] - b[2];
        T dot = move0*T(_out[0]) + move1*T(_out[1]) + move2*T(_out[2]);
        if (dot < T(0)) residual[0] = T(_w) * (-dot);
        else residual[0] = T(0);
        return true;
    }
    float _w;
    double _out[3];
    static ceres::CostFunction* Create(const cv::Vec3d& outward, float w = 1.0)
    {
        return new ceres::AutoDiffCostFunction<ForwardProgressLoss, 1, 3, 3>(new ForwardProgressLoss(outward, w));
    }
};

// Discourages the candidate movement from becoming parallel to a reference axis (degenerate frame)
struct MovementOrthogonalityLoss {
    MovementOrthogonalityLoss(const cv::Vec3d& v_ref, double cos_max, float w) : _w(w) {
        double n = std::sqrt(v_ref[0]*v_ref[0] + v_ref[1]*v_ref[1] + v_ref[2]*v_ref[2]);
        if (n > 1e-12) {
            _v[0] = v_ref[0]/n; _v[1] = v_ref[1]/n; _v[2] = v_ref[2]/n;
        } else {
            _v[0] = _v[1] = _v[2] = 0.0;
        }
        _cos_max = cos_max;
    }
    template <typename T>
    bool operator()(const T* const p, const T* const b, T* residual) const {
        T mv0 = p[0] - b[0];
        T mv1 = p[1] - b[1];
        T mv2 = p[2] - b[2];
        T len = ceres::sqrt(mv0*mv0 + mv1*mv1 + mv2*mv2);
        if (len < T(1e-12)) { residual[0] = T(0); return true; }
        T dot_unit = (mv0*T(_v[0]) + mv1*T(_v[1]) + mv2*T(_v[2])) / len;
        T ad = ceres::abs(dot_unit);
        T excess = ad - T(_cos_max);
        if (excess > T(0)) residual[0] = T(_w) * excess; else residual[0] = T(0);
        return true;
    }
    float _w;
    double _v[3];
    double _cos_max;
    static ceres::CostFunction* Create(const cv::Vec3d& v_ref, double cos_max, float w = 1.0)
    {
        return new ceres::AutoDiffCostFunction<MovementOrthogonalityLoss, 1, 3, 3>(new MovementOrthogonalityLoss(v_ref, cos_max, w));
    }
};

// Penalizes flips of the induced normal n_move = normalize((p - b) x v_ref)
// relative to a reference local normal n_ref.
struct NormalFlipLoss {
    NormalFlipLoss(const cv::Vec3d& v_ref, const cv::Vec3d& n_ref, float w) : _w(w) {
        auto norm = [](const cv::Vec3d& v) {
            double n = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
            return n;
        };
        double nv = norm(v_ref);
        double nn = norm(n_ref);
        if (nv > 1e-12) { _v[0] = v_ref[0]/nv; _v[1] = v_ref[1]/nv; _v[2] = v_ref[2]/nv; }
        else { _v[0] = _v[1] = _v[2] = 0.0; }
        if (nn > 1e-12) { _n[0] = n_ref[0]/nn; _n[1] = n_ref[1]/nn; _n[2] = n_ref[2]/nn; }
        else { _n[0] = _n[1] = _n[2] = 0.0; }
    }
    template <typename T>
    bool operator()(const T* const p, const T* const b, T* residual) const {
        T mv0 = p[0] - b[0];
        T mv1 = p[1] - b[1];
        T mv2 = p[2] - b[2];
        T len = ceres::sqrt(mv0*mv0 + mv1*mv1 + mv2*mv2);
        if (len < T(1e-12)) { residual[0] = T(0); return true; }
        // unit movement
        T um0 = mv0/len, um1 = mv1/len, um2 = mv2/len;
        // n_move = um x v_ref
        T nx = um1*T(_v[2]) - um2*T(_v[1]);
        T ny = um2*T(_v[0]) - um0*T(_v[2]);
        T nz = um0*T(_v[1]) - um1*T(_v[0]);
        T nlen = ceres::sqrt(nx*nx + ny*ny + nz*nz);
        if (nlen < T(1e-12)) { residual[0] = T(0); return true; }
        nx /= nlen; ny /= nlen; nz /= nlen;
        T dot = nx*T(_n[0]) + ny*T(_n[1]) + nz*T(_n[2]);
        if (dot < T(0)) residual[0] = T(_w) * (-dot); else residual[0] = T(0);
        return true;
    }
    float _w;
    double _v[3];
    double _n[3];
    static ceres::CostFunction* Create(const cv::Vec3d& v_ref, const cv::Vec3d& n_ref, float w = 1.0)
    {
        return new ceres::AutoDiffCostFunction<NormalFlipLoss, 1, 3, 3>(new NormalFlipLoss(v_ref, n_ref, w));
    }
};

struct StraightLoss {
    StraightLoss(float w) : _w(w) {};
    static constexpr double kStraightAngleCosThreshold = 0.86602540378443864676; // cos(30°); deviations beyond 30° incur penalty
    template <typename T>
    bool operator()(const T* const a, const T* const b, const T* const c, T* residual) const {
        T d1[3], d2[3];
        d1[0] = b[0] - a[0];
        d1[1] = b[1] - a[1];
        d1[2] = b[2] - a[2];
        
        d2[0] = c[0] - b[0];
        d2[1] = c[1] - b[1];
        d2[2] = c[2] - b[2];
        
        T l1 = sqrt(d1[0]*d1[0] + d1[1]*d1[1] + d1[2]*d1[2]);
        T l2 = sqrt(d2[0]*d2[0] + d2[1]*d2[1] + d2[2]*d2[2]);

        // Guard degenerate segments
        if (l1 <= T(1e-12) || l2 <= T(1e-12)) {
            std::cout << "[StraightLoss:degenerate] l1=" << val(l1)
                      << " l2=" << val(l2)
                      << " a=(" << val(a[0]) << "," << val(a[1]) << "," << val(a[2]) << ")"
                      << " b=(" << val(b[0]) << "," << val(b[1]) << "," << val(b[2]) << ")"
                      << " c=(" << val(c[0]) << "," << val(c[1]) << "," << val(c[2]) << ")"
                      << std::endl;
            residual[0] = T(0);
            return true;
        }

        T dot = (d1[0]*d2[0] + d1[1]*d2[1] + d1[2]*d2[2])/(l1*l2);
        
        if (dot <= T(kStraightAngleCosThreshold)) {
            T penalty = T(kStraightAngleCosThreshold)-dot;
            residual[0] = T(_w)*(T(1)-dot) + T(_w*8)*penalty*penalty;
        } else
            residual[0] = T(_w)*(T(1)-dot);

        return true;
    }

    float _w;

    static ceres::CostFunction* Create(float w = 1.0)
    {
        return new ceres::AutoDiffCostFunction<StraightLoss, 1, 3, 3, 3>(new StraightLoss(w));
    }
};

struct StraightLoss2 {
    StraightLoss2(float w) : _w(w) {};
    template <typename T>
    bool operator()(const T* const a, const T* const b, const T* const c, T* residual) const {
        T avg[3];
        avg[0] = (a[0]+c[0])*T(0.5);
        avg[1] = (a[1]+c[1])*T(0.5);
        avg[2] = (a[2]+c[2])*T(0.5);
        
        residual[0] = T(_w)*(b[0]-avg[0]);
        residual[1] = T(_w)*(b[1]-avg[1]);
        residual[2] = T(_w)*(b[2]-avg[2]);
        
        return true;
    }
    
    float _w;
    
    static ceres::CostFunction* Create(float w = 1.0)
    {
        return new ceres::AutoDiffCostFunction<StraightLoss2, 3, 3, 3, 3>(new StraightLoss2(w));
    }
};

struct StraightLoss2D {
    StraightLoss2D(float w) : _w(w) {};
    template <typename T>
    bool operator()(const T* const a, const T* const b, const T* const c, T* residual) const {
        T d1[2], d2[2];
        d1[0] = b[0] - a[0];
        d1[1] = b[1] - a[1];

        d2[0] = c[0] - b[0];
        d2[1] = c[1] - b[1];

        T l1 = sqrt(d1[0]*d1[0] + d1[1]*d1[1]);
        T l2 = sqrt(d2[0]*d2[0] + d2[1]*d2[1]);

        if (l1 <= T(0) || l2 <= T(0)) {
            residual[0] = T(_w)*((d1[0]*d1[0] + d1[1]*d1[1])*(d2[0]*d2[0] + d2[1]*d2[1]) - T(1));
            std::cout << "uhohh2" << std::endl;
            return true;
        }

        T dot = (d1[0]*d2[0] + d1[1]*d2[1])/(l1*l2);

        residual[0] = T(_w)*(T(1)-dot);

        return true;
    }

    float _w;

    static ceres::CostFunction* Create(float w = 1.0)
    {
        return new ceres::AutoDiffCostFunction<StraightLoss2D, 1, 2, 2, 2>(new StraightLoss2D(w));
    }
};

template<typename T, typename E, int C>
void interp_lin_2d(const cv::Mat_<cv::Vec<E,C>> &m, const T &y, const T &x, T *v) {
    int yi = val(y);
    int xi = val(x);

    T fx = x - T(xi);
    T fy = y - T(yi);

    cv::Vec<E,C> c00 = m(yi,xi);
    cv::Vec<E,C> c01 = m(yi,xi+1);
    cv::Vec<E,C> c10 = m(yi+1,xi);
    cv::Vec<E,C> c11 = m(yi+1,xi+1);

    for (int i=0;i<C;i++) {
        T c0 = (T(1)-fx)*T(c00[i]) + fx*T(c01[i]);
        T c1 = (T(1)-fx)*T(c10[i]) + fx*T(c11[i]);
        v[i] = (T(1)-fy)*c0 + fy*c1;
    }
}

template<typename E1, typename E2, int C>
cv::Vec<E2,C> interp_lin_2d(const cv::Mat_<cv::Vec<E2,C>> &m, const cv::Vec<E1,2> &l)
{
    cv::Vec<E1,C> v;
    interp_lin_2d(m, l[0], l[1], &v[0]);
    return v;
}

struct SurfaceLossD {
    //NOTE we expect loc to be [y, x]
    SurfaceLossD(const cv::Mat_<cv::Vec3f> &m, float w) : _m(m), _w(w) {};
    template <typename T>
    bool operator()(const T* const p, const T* const l, T* residual) const {
        T v[3];

        if (!loc_valid(_m, {val(l[0]), val(l[1])})) {
            residual[0] = T(0);
            residual[1] = T(0);
            residual[2] = T(0);
            return true;
        }

        interp_lin_2d(_m, l[0], l[1], v);

        residual[0] = T(_w)*(v[0] - p[0]);
        residual[1] = T(_w)*(v[1] - p[1]);
        residual[2] = T(_w)*(v[2] - p[2]);

        return true;
    }

    const cv::Mat_<cv::Vec3f> _m;
    float _w;

    static ceres::CostFunction* Create(const cv::Mat_<cv::Vec3f> &m, float w = 1.0)
    {
        return new ceres::AutoDiffCostFunction<SurfaceLossD, 3, 3, 2>(new SurfaceLossD(m, w));
    }

};

struct LinChkDistLoss {
    LinChkDistLoss(const cv::Vec2d &p, float w) : _p(p), _w(w) {};
    template <typename T>
    bool operator()(const T* const p, T* residual) const {
        T a = abs(p[0]-T(_p[0]));
        T b = abs(p[1]-T(_p[1]));
        if (a > T(0))
            residual[0] = T(_w)*sqrt(a);
        else
            residual[0] = T(0);

        if (b > T(0))
            residual[1] = T(_w)*sqrt(b);
        else
            residual[1] = T(0);

        return true;
    }

    cv::Vec2d _p;
    float _w;

    static ceres::CostFunction* Create(const cv::Vec2d &p, float w = 1.0)
    {
        return new ceres::AutoDiffCostFunction<LinChkDistLoss, 2, 2>(new LinChkDistLoss(p, w));
    }

};

struct ZCoordLoss {
    ZCoordLoss(float z, float w) :  _z(z), _w(w) {};
    template <typename T>
    bool operator()(const T* const p, T* residual) const {
        residual[0] = T(_w)*(p[2] - T(_z));
        
        return true;
    }
    
    float _z;
    float _w;
    
    static ceres::CostFunction* Create(float z, float w = 1.0)
    {
        return new ceres::AutoDiffCostFunction<ZCoordLoss, 1, 3>(new ZCoordLoss(z, w));
    }
    
};

template <typename V>
struct ZLocationLoss {
    ZLocationLoss(const cv::Mat_<V> &m, float z, float w) :  _m(m), _z(z), _w(w) {};
    template <typename T>
    bool operator()(const T* const l, T* residual) const {
        T p[3];
        
        if (!loc_valid(_m, {val(l[0]), val(l[1])})) {
            residual[0] = T(0);
            return true;
        }
        
        interp_lin_2d(_m, l[0], l[1], p);
        
        residual[0] = T(_w)*(p[2] - T(_z));
        
        return true;
    }
    
    const cv::Mat_<V> _m;
    float _z;
    float _w;
    
    static ceres::CostFunction* Create(const cv::Mat_<V> &m, float z, float w = 1.0)
    {
        return new ceres::AutoDiffCostFunction<ZLocationLoss, 1, 2>(new ZLocationLoss(m, z, w));
    }
    
};

template <typename T, typename C>
struct SpaceLossAcc {
    SpaceLossAcc(Chunked3d<T,C> &t, float w) : _interpolator(std::make_unique<CachedChunked3dInterpolator<T,C>>(t)), _w(w) {};
    template <typename E>
    bool operator()(const E* const l, E* residual) const {
        E v;

        _interpolator->template Evaluate<E>(l[2], l[1], l[0], &v);

        residual[0] = E(_w)*v;

        return true;
    }

    float _w;
    std::unique_ptr<CachedChunked3dInterpolator<T,C>> _interpolator;

    static ceres::CostFunction* Create(Chunked3d<T,C> &t, float w = 1.0)
    {
        return new ceres::AutoDiffCostFunction<SpaceLossAcc<T,C>, 1, 3>(new SpaceLossAcc<T,C>(t, w));
    }

};

template <typename T, typename C>
struct SpaceLineLossAcc {
    SpaceLineLossAcc(Chunked3d<T,C> &t, int steps, float w) : _steps(steps), _w(w)
    {
        _interpolator.resize(_steps-1);
        for(int i=1;i<_steps;i++)
            _interpolator[i-1].reset(new CachedChunked3dInterpolator<T,C>(t));
    };
    template <typename E>
    bool operator()(const E* const la, const E* const lb, E* residual) const {
        E v;
        E sum = E(0);

        bool ign = false;

        for(int i=1;i<_steps;i++) {
            E f2 = E(float(i)/_steps);
            E f1 = E(1.0f-float(i)/_steps);
            _interpolator[i-1].get()->template Evaluate<E>(f1*la[2]+f2*lb[2], f1*la[1]+f2*lb[1], f1*la[0]+f2*lb[0], &v);
            sum += E(_w)*v;
        }

        residual[0] = sum/E(_steps-1);

        return true;
    }

    std::vector<std::unique_ptr<CachedChunked3dInterpolator<T,C>>> _interpolator;
    int _steps;
    float _w;

    static ceres::CostFunction* Create(Chunked3d<T,C> &t, int steps, float w = 1.0)
    {
        return new ceres::AutoDiffCostFunction<SpaceLineLossAcc, 1, 3, 3>(new SpaceLineLossAcc(t, steps, w));
    }

};

struct FiberDirectionLoss {
    FiberDirectionLoss(Chunked3dVec3fFromUint8 &fiber_dirs, Chunked3dFloatFromUint8 *maybe_weights, float w) :
        _fiber_dirs(fiber_dirs), _maybe_weights(maybe_weights), _w(w) {};
    template <typename E>
    bool operator()(const E* const l_base, const E* const l_u_off, E* residual) const {

        // Both l_base and l_u_off are indexed xyz!

        // Note this does *not* sample the direction volume differentiably. This makes sense for now since the volume
        // is piecewise constant, and interpolating it is non-trivial anyway (since its values live in RP2)
        cv::Vec3f fiber_dir_zyx_vec = _fiber_dirs(unjet(l_base[2]), unjet(l_base[1]), unjet(l_base[0]));
        E fiber_dir_zyx[3] = {E(fiber_dir_zyx_vec[0]), E(fiber_dir_zyx_vec[1]), E(fiber_dir_zyx_vec[2])};

        E const patch_u_disp_zyx[3] {
            l_u_off[2] - l_base[2],
            l_u_off[1] - l_base[1],
            l_u_off[0] - l_base[0],
        };

        // fiber_dir is now a unit vector in zyx order, pointing along our fibers (so in U-/V-direction of patch)
        // l_u_off is assumed to be the location for a 2D point that is shifted along the U-/V-direction from l_base
        // patch_u_disp is the displacement between l_base and l_u_off, which we want to be aligned with the fiber direction, modulo flips

        E const patch_u_dist = sqrt(patch_u_disp_zyx[0] * patch_u_disp_zyx[0]
                                    + patch_u_disp_zyx[1] * patch_u_disp_zyx[1]
                                    + patch_u_disp_zyx[2] * patch_u_disp_zyx[2]
                                    + E(1e-12));
        E const abs_dot = abs(patch_u_disp_zyx[0] * fiber_dir_zyx[0]
                              + patch_u_disp_zyx[1] * fiber_dir_zyx[1]
                              + patch_u_disp_zyx[2] * fiber_dir_zyx[2]) / patch_u_dist;

        E const weight_at_point = _maybe_weights ? E((*_maybe_weights)(unjet(l_base[2]), unjet(l_base[1]), unjet(l_base[0]))) : E(1);

        residual[0] = E(_w) * (E(1) - abs_dot) * weight_at_point;

        return true;
    }

    static double unjet(const double& v) { return v; }
    template<typename JetT> static double unjet(const JetT& v) { return v.a; }

    float _w;
    Chunked3dVec3fFromUint8 &_fiber_dirs;
    Chunked3dFloatFromUint8 *_maybe_weights;

    static ceres::CostFunction* Create(Chunked3dVec3fFromUint8 &fiber_dirs, Chunked3dFloatFromUint8 *maybe_weights, float w = 1.0)
    {
        return new ceres::AutoDiffCostFunction<FiberDirectionLoss, 1, 3, 3>(new FiberDirectionLoss(fiber_dirs, maybe_weights, w));
    }
};

struct NormalDirectionLoss {
    NormalDirectionLoss(Chunked3dVec3fFromUint8 &normal_dirs, Chunked3dFloatFromUint8 *maybe_weights, float w) :
        _normal_dirs(normal_dirs), _maybe_weights(maybe_weights), _w(w) {};
    template <typename E>
    bool operator()(const E* const l_base, const E* const l_u_off, const E* const l_v_off, E* residual) const {

        // All l_* are indexed xyz, while fiber_field zarr _normal_dirs is indexed zyx (and contains zyx-ordered vectors)

        // Note this does *not* sample the direction volume differentiably, i.e. there is a gradient moving the points to
        // be more-normal, but not moving the surface to be in a region where the normal field better matches the current
        // surface orientation
        cv::Vec3f target_normal_zyx_vec = _normal_dirs(unjet(l_base[2]), unjet(l_base[1]), unjet(l_base[0]));
        E target_normal_zyx[3] = {E(target_normal_zyx_vec[0]), E(target_normal_zyx_vec[1]), E(target_normal_zyx_vec[2])};

        E const patch_u_disp_zyx[3] {
            l_u_off[2] - l_base[2],
            l_u_off[1] - l_base[1],
            l_u_off[0] - l_base[0],
        };
        E const patch_v_disp_zyx[3] {
            l_v_off[2] - l_base[2],
            l_v_off[1] - l_base[1],
            l_v_off[0] - l_base[0],
        };

        // target_normal_zyx is a unit vector, hopefully pointing normal to the surface
        // patch_*_disp are horizontal and vertical displacements in the surface plane, tangent at l_base
        // patch_normal_zyx will be the cross of the above, i.e. actual normal of the surface
        // we want patch_normal and target_normal to be aligned, modulo flips

        E const patch_normal_zyx[3] {
            patch_u_disp_zyx[1] * patch_v_disp_zyx[2] - patch_u_disp_zyx[2] * patch_v_disp_zyx[1],
            patch_u_disp_zyx[2] * patch_v_disp_zyx[0] - patch_u_disp_zyx[0] * patch_v_disp_zyx[2],
            patch_u_disp_zyx[0] * patch_v_disp_zyx[1] - patch_u_disp_zyx[1] * patch_v_disp_zyx[0],
        };
        E const patch_normal_length = sqrt(patch_normal_zyx[0] * patch_normal_zyx[0] + patch_normal_zyx[1] * patch_normal_zyx[1] + patch_normal_zyx[2] * patch_normal_zyx[2]);
        if (patch_normal_length < E(1e-12)) {
            std::cout << "[NormalDir:zero_normal] len=" << val(patch_normal_length)
                      << " base=(" << val(l_base[0]) << "," << val(l_base[1]) << "," << val(l_base[2]) << ")"
                      << " u_off=(" << val(l_u_off[0]) << "," << val(l_u_off[1]) << "," << val(l_u_off[2]) << ")"
                      << " v_off=(" << val(l_v_off[0]) << "," << val(l_v_off[1]) << "," << val(l_v_off[2]) << ")"
                      << std::endl;
            residual[0] = E(0);
            return true;
        }

        E const dot_raw_nd = patch_normal_zyx[0] * target_normal_zyx[0] + patch_normal_zyx[1] * target_normal_zyx[1] + patch_normal_zyx[2] * target_normal_zyx[2];
        // [DISABLED] abs smoothing and log; revert to exact absolute value
        // if (ceres::abs(dot_raw_nd) < E(1e-6)) {
        //     std::cout << "[NormalDir:abs_smooth] dot_raw=" << val(dot_raw_nd) << std::endl;
        // }
        E const abs_dot = ceres::abs(dot_raw_nd) / patch_normal_length;

        E const weight_at_point = _maybe_weights ? E((*_maybe_weights)(unjet(l_base[2]), unjet(l_base[1]), unjet(l_base[0]))) : E(1);

        residual[0] = E(_w) * (E(1) - abs_dot) * weight_at_point;

        return true;
    }

    static double unjet(const double& v) { return v; }
    template<typename JetT> static double unjet(const JetT& v) { return v.a; }

    float _w;
    Chunked3dVec3fFromUint8 &_normal_dirs;
    Chunked3dFloatFromUint8 *_maybe_weights;

    static ceres::CostFunction* Create(Chunked3dVec3fFromUint8 &normal_dirs, Chunked3dFloatFromUint8 *maybe_weights, float w = 1.0)
    {
        return new ceres::AutoDiffCostFunction<NormalDirectionLoss, 1, 3, 3, 3>(new NormalDirectionLoss(normal_dirs, maybe_weights, w));
    }
};

/**
 * @brief Ceres cost function to enforce that the surface normal aligns with precomputed normal grids.
 *
 * The loss is applied per corner of each quad and per Cartesian plane (XY, XZ, YZ).
 * For each quad (A, B1, B2, C), the loss is calculated relative to an imaginary plane P
 * that is one of the Cartesian planes shifted to pass through the base point A.
 *
 * 1.  The loss is skipped (residual set to 0) if all points of the quad lie on the same side of P.
 * 2.  The side of the opposing point C relative to P determines which side point (B1 or B2) is used.
 *     We select the side point Bn that is on the opposite side of P from C.
 * 3.  The intersection point E of the line segment C-Bn with the plane P is calculated.
 * 4.  The loss is then computed using the 2D normal constraint logic between the projected point A
 *     (with the coordinate defining P removed) and the projected intersection point E.
 * 5.  The final residual is weighted by the angle between the plane defined by (A, Bn, C) and
 *     the Cartesian plane P. The weight is 1 for a 90-degree angle and 0 for a 0-degree angle.
 */
struct NormalConstraintPlane {
    const vc::core::util::NormalGridVolume& normal_grid_volume;
    const int plane_idx; // 0: XY, 1: XZ, 2: YZ
    const double w_normal;
    const double w_snap;
    const int z_min;
    const int z_max;
    bool invert_dir;
 
     template<typename T>
     struct PointPairCache {
         cv::Point2f p1_ = {-1, -1}, p2_ = {-1, -1};
         T payload_;
         // Cache must be tied to the specific GridStore slice, not the volume.
         const vc::core::util::GridStore* grid_source = nullptr;
         // Legacy identity (pre-fix) used the volume pointer; keep for debug comparison.
         const vc::core::util::NormalGridVolume* legacy_grid_source_volume = nullptr;
 
         bool valid(const cv::Point2f& p1, const cv::Point2f& p2, float th, const vc::core::util::GridStore* current_grid_source) const {
             if (grid_source != current_grid_source) return false;
             cv::Point2f d1 = p1 - p1_;
             cv::Point2f d2 = p2 - p2_;
             return (d1.dot(d1) + d2.dot(d2)) < th * th;
         }
 
         const T& get() const { return payload_; }
 
         void put(const cv::Point2f& p1, const cv::Point2f& p2, T payload, const vc::core::util::GridStore* new_grid_source) {
             p1_ = p1;
             p2_ = p2;
             payload_ = std::move(payload);
             grid_source = new_grid_source;
         }
     };
 
     // Caching for nearby paths
     using PathCachePayload = std::vector<std::vector<cv::Point>>;
     mutable std::array<PointPairCache<PathCachePayload>, 2> path_caches_;
 
     // Caching for snapping loss results
     struct SnapLossPayload {
         int best_path_idx = -1;
         int best_seg_idx = -1;
         bool best_is_next = false;
     };
     mutable std::array<PointPairCache<SnapLossPayload>, 2> snap_loss_caches_;
 
     const float cache_radius_normal_ = 16.0f;
     const float cache_radius_snap_ = 1.0f;
     const float roi_radius_ = 64.0f;
     const float query_radius_ = roi_radius_ + 16.0f;
     const double snap_trig_th_ = 4.0;
     const double snap_search_range_ = 16.0;
 
     NormalConstraintPlane(const vc::core::util::NormalGridVolume& normal_grid_volume, int plane_idx, double w_normal, double w_snap, bool direction_aware = false, int z_min = -1, int z_max = -1, bool invert_dir = false)
         : normal_grid_volume(normal_grid_volume), plane_idx(plane_idx), w_normal(w_normal), w_snap(w_snap), direction_aware_(direction_aware), z_min(z_min), z_max(z_max), invert_dir(invert_dir) {}

    template <typename T>
    bool operator()(const T* const pA, const T* const pB1, const T* const pB2, const T* const pC, T* residual) const {
        residual[0] = T(0.0);

        // Use consistent XYZ indexing. plane_idx 0=XY (normal Z), 1=XZ (normal Y), 2=YZ (normal X)
        int normal_axis = 2 - plane_idx;
        T a_coord = pA[normal_axis];

        T b1_rel = pB1[normal_axis] - a_coord;
        T b2_rel = pB2[normal_axis] - a_coord;
        T c_rel = pC[normal_axis] - a_coord;

        // Skip if all points are on the same side of the plane P through A.
        if ((b1_rel > T(0) && b2_rel > T(0) && c_rel > T(0)) ||
            (b1_rel < T(0) && b2_rel < T(0) && c_rel < T(0))) {
            return true;
        }

        const T* pBn = nullptr;
        T bn_rel;

        // Choose Bn.
        if (ceres::abs(c_rel) < T(1e-9)) { // If C is on the plane...
            // ...choose a B that is not on the plane.
            //TODO choose the large one!
            if (ceres::abs(b1_rel) > T(1e-9)) { pBn = pB1; bn_rel = b1_rel; }
            else if (ceres::abs(b2_rel) > T(1e-9)) { pBn = pB2; bn_rel = b2_rel; }
        } else { // If C is not on the plane...
            // ...choose a B on the opposite side of C (inclusive).
            if (c_rel > T(0)) {
                if (b1_rel <= T(0)) { pBn = pB1; bn_rel = b1_rel; }
                else if (b2_rel <= T(0)) { pBn = pB2; bn_rel = b2_rel; }
            } else { // c_rel < T(0)
                if (b1_rel >= T(0)) { pBn = pB1; bn_rel = b1_rel; }
                else if (b2_rel >= T(0)) { pBn = pB2; bn_rel = b2_rel; }
            }
        }

        if (pBn == nullptr) {
            std::cout << "[NCP:no_Bn] plane=" << plane_idx
                      << " b1_rel=" << val(b1_rel)
                      << " b2_rel=" << val(b2_rel)
                      << " c_rel=" << val(c_rel)
                      << " A=(" << val(pA[0]) << "," << val(pA[1]) << "," << val(pA[2]) << ")"
                      << " B1=(" << val(pB1[0]) << "," << val(pB1[1]) << "," << val(pB1[2]) << ")"
                      << " B2=(" << val(pB2[0]) << "," << val(pB2[1]) << "," << val(pB2[2]) << ")"
                      << " C=(" << val(pC[0]) << "," << val(pC[1]) << "," << val(pC[2]) << ")"
                      << std::endl;
            return true; // C is on the plane, or B1/B2 are on the same side as C.
        }

        // Intersection of segment C-Bn with plane P.
        T denominator = bn_rel - c_rel;
        if (ceres::abs(denominator) < T(1e-9)) {
            // Log when the segment C-Bn is (near) parallel to the plane through A on the chosen axis.
            std::cout << "[NCP:parallel] plane=" << plane_idx
                      << " denom=" << val(denominator)
                      << " bn_rel=" << val(bn_rel)
                      << " c_rel=" << val(c_rel)
                      << " A=(" << val(pA[0]) << "," << val(pA[1]) << "," << val(pA[2]) << ")"
                      << " C=(" << val(pC[0]) << "," << val(pC[1]) << "," << val(pC[2]) << ")"
                      << std::endl;
            return true; // Avoid division by zero if segment is parallel to plane.
        }
        T t = -c_rel / denominator;
        T pE[3];
        for (int i = 0; i < 3; ++i) {
            pE[i] = pC[i] + t * (pBn[i] - pC[i]);
        }

        // Project A and E onto the 2D plane.
        T pA_2d[2], pE_2d[2];
        int coord_idx = 0;
        if (plane_idx == 0) { // XY plane
            pA_2d[0] = pA[0]; pA_2d[1] = pA[1];
            pE_2d[0] = pE[0]; pE_2d[1] = pE[1];
        } else if (plane_idx == 1) { // XZ plane
            pA_2d[0] = pA[0]; pA_2d[1] = pA[2];
            pE_2d[0] = pE[0]; pE_2d[1] = pE[2];
        } else { // YZ plane
            pA_2d[0] = pA[1]; pA_2d[1] = pA[2];
            pE_2d[0] = pE[1]; pE_2d[1] = pE[2];
        }

        // Query the normal grids.
        //FIXME query in middle!
        cv::Point3f query_point(val(pA[0]), val(pA[1]), val(pA[2]));

        if (z_min != -1 && query_point.z < z_min) return true;
        if (z_max != -1 && query_point.z > z_max) return true;

        // [DISABLED] Blended slice smoothing across adjacent grids; revert to nearest grid behavior
        // auto grid_query = normal_grid_volume.query(query_point, plane_idx);
        // if (!grid_query) { return true; }
        // T loss1 = invert_dir
        //           ? calculate_normal_snapping_loss(pE_2d, pA_2d, *grid_query->grid1, 0)
        //           : calculate_normal_snapping_loss(pA_2d, pE_2d, *grid_query->grid1, 0);
        // T loss2 = invert_dir
        //           ? calculate_normal_snapping_loss(pE_2d, pA_2d, *grid_query->grid2, 1)
        //           : calculate_normal_snapping_loss(pA_2d, pE_2d, *grid_query->grid2, 1);
        // T interpolated_loss = (T(1.0) - T(grid_query->weight)) * loss1 + T(grid_query->weight) * loss2;
        auto grid_ptr = normal_grid_volume.query_nearest(query_point, plane_idx);
        if (!grid_ptr) {
            return true;
        }
        T interpolated_loss;
        if (invert_dir)
            interpolated_loss = calculate_normal_snapping_loss(pE_2d, pA_2d, *grid_ptr, 0);
        else
            interpolated_loss = calculate_normal_snapping_loss(pA_2d, pE_2d, *grid_ptr, 0);

        // Calculate angular weight.
        double v_abn[3], v_ac[3];
        for(int i=0; i<3; ++i) {
            v_abn[i] = val(pBn[i]) - val(pA[i]);
            v_ac[i] = val(pC[i]) - val(pA[i]);
        }

        double cross_product[3] = {
            v_abn[1] * v_ac[2] - v_abn[2] * v_ac[1],
            v_abn[2] * v_ac[0] - v_abn[0] * v_ac[2],
            v_abn[0] * v_ac[1] - v_abn[1] * v_ac[0]
        };

        double cross_len = std::sqrt(cross_product[0]*cross_product[0] + cross_product[1]*cross_product[1] + cross_product[2]*cross_product[2]);
        double plane_normal_coord = cross_product[normal_axis];
        
        if (cross_len < 1e-9) {
            std::cout << "[NCP:angle_degenerate] plane=" << plane_idx
                      << " cross_len=" << cross_len << std::endl;
            return true;
        }

        double cos_angle_raw = plane_normal_coord / (cross_len + 1e-9);
        double cos_angle = cos_angle_raw;
        // Clamp to valid cosine range to guard numerical drift; log if clamped
        if (cos_angle > 1.0) {
            std::cout << "[NCP:angle_clamp] plane=" << plane_idx
                      << " cos_raw=" << cos_angle_raw << std::endl;
            cos_angle = 1.0;
        }
        if (cos_angle < -1.0) {
            std::cout << "[NCP:angle_clamp] plane=" << plane_idx
                      << " cos_raw=" << cos_angle_raw << std::endl;
            cos_angle = -1.0;
        }
        double angle_weight = 1.0 - abs(cos_angle) ;// * cos_angle; // sin^2(angle)
        //good but slow?
        // double angle_weight = sqrt(1.0 - abs(cos_angle)+1e-9); // * cos_angle; // sin^2(angle)

        residual[0] = interpolated_loss * T(angle_weight);

        return true;
    }

    static float point_line_dist_sq(const cv::Point2f& p, const cv::Point2f& a, const cv::Point2f& b) {
        cv::Point2f ab = b - a;
        cv::Point2f ap = p - a;
        float ab_len_sq = ab.dot(ab);
        if (ab_len_sq < 1e-9) {
            return ap.dot(ap);
        }
        float t = ap.dot(ab) / ab_len_sq;
        t = std::max(0.0f, std::min(1.0f, t));
        cv::Point2f projection = a + t * ab;
        return (p - projection).dot(p - projection);
    }

    template <typename T>
    static T point_line_dist_sq_differentiable(const T* p, const cv::Point2f& a, const cv::Point2f& b) {
        T ab_x = T(b.x - a.x);
        T ab_y = T(b.y - a.y);
        T ap_x = p[0] - T(a.x);
        T ap_y = p[1] - T(a.y);

        T ab_len_sq = ab_x * ab_x + ab_y * ab_y;
        if (ab_len_sq < T(1e-9)) {
            static int s_seg_deg_log_count = 0;
            if ((++s_seg_deg_log_count % 5) == 0) {
                std::cout << "[NCP:seg_degenerate] a=(" << a.x << "," << a.y << ")"
                          << " b=(" << b.x << "," << b.y << ")"
                          << " p=(" << val(p[0]) << "," << val(p[1]) << ")"
                          << std::endl;
            }
            return ap_x * ap_x + ap_y * ap_y;
        }
        T t = (ap_x * ab_x + ap_y * ab_y) / ab_len_sq;

        // Clamping t using conditionals that are safe for Jets
        if (t < T(0.0)) t = T(0.0);
        if (t > T(1.0)) t = T(1.0);

        T proj_x = T(a.x) + t * ab_x;
        T proj_y = T(a.y) + t * ab_y;

        T dx = p[0] - proj_x;
        T dy = p[1] - proj_y;
        return dx * dx + dy * dy;
    }

    PathCachePayload filter_and_split_paths(const std::vector<std::shared_ptr<std::vector<cv::Point>>>& raw_paths, const cv::Point2f& p1, const cv::Point2f& p2) const {
        PathCachePayload filtered_paths;
        for (const std::shared_ptr<std::vector<cv::Point>>& path_ptr : raw_paths) {
            const std::vector<cv::Point>& path = *path_ptr;
            if (path.size() < 2) continue;

            std::vector<cv::Point> current_sub_path;
            for (size_t i = 0; i < path.size() - 1; ++i) {
                const cv::Point2f& p_a = path[i];
                const cv::Point2f& p_b = path[i+1];

                if (seg_dist_sq_appx(p1, p2, p_a, p_b) <= query_radius_ * query_radius_) {
                    if (current_sub_path.empty()) {
                        current_sub_path.push_back(p_a);
                    }
                    current_sub_path.push_back(p_b);
                } else {
                    if (current_sub_path.size() > 1) {
                        filtered_paths.push_back(current_sub_path);
                    }
                    current_sub_path.clear();
                }
            }
            if (current_sub_path.size() > 1) {
                filtered_paths.push_back(current_sub_path);
            }
        }
        return filtered_paths;
    }

    template <typename T>
    T calculate_normal_snapping_loss(const T* p1, const T* p2, const vc::core::util::GridStore& normal_grid, int grid_idx) const {
        T edge_vec_x = p2[0] - p1[0];
        T edge_vec_y = p2[1] - p1[1];

        T edge_len_sq = edge_vec_x * edge_vec_x + edge_vec_y * edge_vec_y;
        T edge_len = ceres::sqrt(edge_len_sq);
        if (edge_len_sq < T(1e-12)) {
            static int s_edge_zero_log_count = 0;
            if ((++s_edge_zero_log_count % 5) == 0) {
                std::cout << "[NCP:edge_zero] plane=" << plane_idx
                          << " grid_idx=" << grid_idx
                          << " edge_len_sq=" << val(edge_len_sq)
                          << " p1=(" << val(p1[0]) << "," << val(p1[1]) << ")"
                          << " p2=(" << val(p2[0]) << "," << val(p2[1]) << ")"
                          << std::endl;
            }
            std::cout << "[NCP:edge_zero_summary] plane=" << plane_idx
                      << " grid_idx=" << grid_idx
                      << " skipped=" << 1
                      << " total=" << 0
                      << std::endl;
            return T(0.0);
        }

        T edge_normal_x = edge_vec_y / edge_len;
        T edge_normal_y = -edge_vec_x / edge_len;

        cv::Point2f p1_cv(val(p1[0]), val(p1[1]));
        cv::Point2f p2_cv(val(p2[0]), val(p2[1]));

        auto& path_cache = path_caches_[grid_idx];

        // Debug: Log only when new-vs-old cache validity differs
        // #if 0
        // {
        //     cv::Point2f d1 = p1_cv - path_cache.p1_;
        //     cv::Point2f d2 = p2_cv - path_cache.p2_;
        //     float disp2 = d1.dot(d1) + d2.dot(d2);
        //     bool new_valid = path_cache.valid(p1_cv, p2_cv, cache_radius_normal_, &normal_grid);
        //     bool old_valid = (path_cache.legacy_grid_source_volume == &normal_grid_volume) && (disp2 < cache_radius_normal_ * cache_radius_normal_);
        //     if (new_valid != old_valid) {
        //         std::cout << "[NormalConstraintPlane:path] plane=" << plane_idx
        //                   << " grid_idx=" << grid_idx
        //                   << " new_gridstore=" << (const void*)(&normal_grid)
        //                   << " old_volume=" << (const void*)(&normal_grid_volume)
        //                   << " disp2=" << disp2
        //                   << " th2=" << (cache_radius_normal_ * cache_radius_normal_)
        //                   << " new_valid=" << new_valid
        //                   << " old_valid=" << old_valid
        //                   << std::endl;
        //     }
        // }
        // #endif

        if (!path_cache.valid(p1_cv, p2_cv, cache_radius_normal_, &normal_grid)) {
            cv::Point2f midpoint_cv = 0.5f * (p1_cv + p2_cv);
            auto raw_paths = normal_grid.get(midpoint_cv, query_radius_);
            auto filtered_paths = filter_and_split_paths(raw_paths, p1_cv, p2_cv);
            path_cache.put(p1_cv, p2_cv, std::move(filtered_paths), &normal_grid);
            path_cache.legacy_grid_source_volume = &normal_grid_volume;
            // Invalidate snapping cache whenever the path cache is updated
            snap_loss_caches_[grid_idx] = {};
        }
        const PathCachePayload& nearby_paths = path_cache.get();

        if (nearby_paths.empty()) {
            return T(0.0);
        }

        T total_weighted_dot_loss = T(0.0);
        T total_weight = T(0.0);
        // int near_tangent_skipped = 0; // [DISABLED] near-tangent guard regressed functionality
        // int segments_considered = 0;  // [DISABLED]

        for (const std::vector<cv::Point>& path : nearby_paths) {
            if (path.size() < 2) continue;

            for (size_t i = 0; i < path.size() - 1; ++i) {
                cv::Point2f p_a = path[i];
                cv::Point2f p_b = path[i+1];

                float dist_sq = seg_dist_sq_appx(p1_cv, p2_cv, p_a, p_b);
                if (dist_sq > roi_radius_*roi_radius_)
                    continue;

                dist_sq = std::max(10.0f, dist_sq);

                T weight_n = T(1.0 / dist_sq);

                cv::Point2f tangent = p_b - p_a;
                float length = cv::norm(tangent);
                if (length > 0) {
                    tangent /= length;
                }
                cv::Point2f normal(-tangent.y, tangent.x);

                // Compute alignment between edge normal and path normal
                // segments_considered++; // [DISABLED]
                T dot_raw = edge_normal_x * T(normal.x) + edge_normal_y * T(normal.y);
                // [DISABLED] near-tangent guard; keep contribution but without smoothing
                // if (ceres::abs(dot_raw) < T(1e-6)) { ... }

                T dot_product = dot_raw;
                if (!direction_aware_) {
                    // [DISABLED] abs smoothing; revert to exact absolute value
                    // dot_product = ceres::sqrt(dot_raw * dot_raw + T(1e-12));
                    dot_product = ceres::abs(dot_raw);
                }

                total_weighted_dot_loss += weight_n*(T(1.0) - dot_product);
                total_weight += weight_n;
            }
        }

        // [DISABLED] near-tangent summary telemetry
        // if (near_tangent_skipped > 0) {
        //     std::cout << "[NCP:near_tangent_summary] plane=" << plane_idx
        //               << " grid_idx=" << grid_idx
        //               << " skipped=" << near_tangent_skipped
        //               << " total=" << segments_considered
        //               << std::endl;
        // }

        T normal_loss = T(0.0);
        if (total_weight > T(1e-9)) {
            normal_loss = total_weighted_dot_loss/total_weight;
        }

        // Snapping logic

        if (w_snap == 0.0)
            return T(w_normal)*normal_loss;

        auto& snap_cache = snap_loss_caches_[grid_idx];
        // Debug: Log only when new-vs-old cache validity differs
        // #if 0
        // {
        //     cv::Point2f d1s = p1_cv - snap_cache.p1_;
        //     cv::Point2f d2s = p2_cv - snap_cache.p2_;
        //     float disp2s = d1s.dot(d1s) + d2s.dot(d2s);
        //     bool new_valid_s = snap_cache.valid(p1_cv, p2_cv, cache_radius_snap_, &normal_grid);
        //     bool old_valid_s = (snap_cache.legacy_grid_source_volume == &normal_grid_volume) && (disp2s < cache_radius_snap_ * cache_radius_snap_);
        //     if (new_valid_s != old_valid_s) {
        //         std::cout << "[NormalConstraintPlane:snap] plane=" << plane_idx
        //                   << " grid_idx=" << grid_idx
        //                   << " new_gridstore=" << (const void*)(&normal_grid)
        //                   << " old_volume=" << (const void*)(&normal_grid_volume)
        //                   << " disp2=" << disp2s
        //                   << " th2=" << (cache_radius_snap_ * cache_radius_snap_)
        //                   << " new_valid=" << new_valid_s
        //                   << " old_valid=" << old_valid_s
        //                   << std::endl;
        //     }
        // }
        // #endif

        if (!snap_cache.valid(p1_cv, p2_cv, cache_radius_snap_, &normal_grid)) {
            SnapLossPayload payload;
            float closest_dist_norm = std::numeric_limits<float>::max();
 
             for (int path_idx = 0; path_idx < nearby_paths.size(); ++path_idx) {
                 const auto& path = nearby_paths[path_idx];
                 if (path.size() < 2) continue;
 
                 for (int i = 0; i < path.size() - 1; ++i) {
                     float d2_sq = point_line_dist_sq(p2_cv, path[i], path[i+1]);
                     if (d2_sq >= snap_trig_th_ * snap_trig_th_) continue;
 
                     if (i < path.size() - 2) { // Check next segment
                         float d1_sq = point_line_dist_sq(p1_cv, path[i+1], path[i+2]);
                         if (d1_sq < snap_search_range_ * snap_search_range_) {
                             float dist_norm = 0.5f * (sqrt(d1_sq)/snap_search_range_ + sqrt(d2_sq)/snap_trig_th_);
                             if (dist_norm < closest_dist_norm) {
                                 closest_dist_norm = dist_norm;
                                 payload.best_path_idx = path_idx;
                                 payload.best_seg_idx = i;
                                 payload.best_is_next = true;
                             }
                         }
                     }
                     if (i > 0) { // Check prev segment
                         float d1_sq = point_line_dist_sq(p1_cv, path[i-1], path[i]);
                          if (d1_sq < snap_search_range_ * snap_search_range_) {
                             float dist_norm = 0.5f * (sqrt(d1_sq)/snap_search_range_ + sqrt(d2_sq)/snap_trig_th_);
                             if (dist_norm < closest_dist_norm) {
                                 closest_dist_norm = dist_norm;
                                 payload.best_path_idx = path_idx;
                                 payload.best_seg_idx = i;
                                 payload.best_is_next = false;
                             }
                         }
                     }
                 }
             }
            snap_cache.put(p1_cv, p2_cv, payload, &normal_grid);
            snap_cache.legacy_grid_source_volume = &normal_grid_volume;
        }

        int seg_deg_skipped_total = 0;
        int seg_deg_checked_total = 0;

        const auto& snap_payload = snap_cache.get();
        T snapping_loss = T(0.0);

        if (snap_payload.best_path_idx != -1) {
            const auto& best_path = nearby_paths[snap_payload.best_path_idx];
            const auto& seg2_p1 = best_path[snap_payload.best_seg_idx];
            const auto& seg2_p2 = best_path[snap_payload.best_seg_idx + 1];

            cv::Point2f seg1_p1, seg1_p2;
            if (snap_payload.best_is_next) {
                seg1_p1 = best_path[snap_payload.best_seg_idx + 1];
                seg1_p2 = best_path[snap_payload.best_seg_idx + 2];
            } else {
                seg1_p1 = best_path[snap_payload.best_seg_idx - 1];
                seg1_p2 = best_path[snap_payload.best_seg_idx];
            }

            auto d_sq_with_deg_check = [&](const T* pp, const cv::Point2f& aa, const cv::Point2f& bb) -> T {
                float abx = bb.x - aa.x;
                float aby = bb.y - aa.y;
                float ab_len_sq_f = abx*abx + aby*aby;
                seg_deg_checked_total++;
                if (ab_len_sq_f < 1e-9f) {
                    static int s_seg_deg_log_count2 = 0;
                    if ((++s_seg_deg_log_count2 % 5) == 0) {
                        std::cout << "[NCP:seg_degenerate] a=(" << aa.x << "," << aa.y << ")"
                                  << " b=(" << bb.x << "," << bb.y << ")"
                                  << " p=(" << val(pp[0]) << "," << val(pp[1]) << ")"
                                  << std::endl;
                    }
                    seg_deg_skipped_total++;
                    T ap_x = pp[0] - T(aa.x);
                    T ap_y = pp[1] - T(aa.y);
                    return ap_x*ap_x + ap_y*ap_y;
                }
                return point_line_dist_sq_differentiable(pp, aa, bb);
            };

            T d1_sq = d_sq_with_deg_check(p1, seg1_p1, seg1_p2);
            T d2_sq = d_sq_with_deg_check(p2, seg2_p1, seg2_p2);

            T d1_norm, d2_norm;
            if (d1_sq < T(1e-9))
                d1_norm = d1_sq / T(snap_search_range_);
            else
                d1_norm = ceres::sqrt(d1_sq) / T(snap_search_range_);

            if (d2_sq < T(1e-9))
                d2_norm = d2_sq / T(snap_trig_th_);
            else
                d2_norm = ceres::sqrt(d2_sq) / T(snap_trig_th_);

            snapping_loss = (d1_norm * (T(1.0) - d2_norm) + d2_norm);
        } else {
            snapping_loss = T(1.0); // Penalty if no snap target found
        }

        if (seg_deg_skipped_total > 0) {
            std::cout << "[NCP:seg_degenerate_summary] plane=" << plane_idx
                      << " grid_idx=" << grid_idx
                      << " skipped=" << seg_deg_skipped_total
                      << " total=" << seg_deg_checked_total
                      << std::endl;
        }

        return T(w_normal)*normal_loss + T(w_snap)*snapping_loss;
    }

    static float seg_dist_sq_appx(cv::Point2f p1, cv::Point2f p2, cv::Point2f p3, cv::Point2f p4)
    {
        cv::Point2f p = 0.5*(p1+p2);
        cv::Point2f s = 0.5*(p3+p4);
        cv::Point2f d = p-s;
        return d.x*d.x+d.y*d.y;
    }

    static float seg_dist_sq(cv::Point2f p1, cv::Point2f p2, cv::Point2f p3, cv::Point2f p4) {
        auto dot = [](cv::Point2f a, cv::Point2f b) { return a.x * b.x + a.y * b.y; };
        auto dist_sq = [&](cv::Point2f p) { return p.x * p.x + p.y * p.y; };
        cv::Point2f u = p2 - p1;
        cv::Point2f v = p4 - p3;
        cv::Point2f w = p1 - p3;
        float a = dot(u, u); float b = dot(u, v); float c = dot(v, v);
        float d = dot(u, w); float e = dot(v, w); float D = a * c - b * b;
        float sc, sN, sD = D; float tc, tN, tD = D;
        if (D < 1e-7) { sN = 0.0; sD = 1.0; tN = e; tD = c; }
        else { sN = (b * e - c * d); tN = (a * e - b * d);
            if (sN < 0.0) { sN = 0.0; tN = e; tD = c; }
            else if (sN > sD) { sN = sD; tN = e + b; tD = c; }
        }
        if (tN < 0.0) { tN = 0.0;
            if (-d < 0.0) sN = 0.0; else if (-d > a) sN = sD;
            else { sN = -d; sD = a; }
        } else if (tN > tD) { tN = tD;
            if ((-d + b) < 0.0) sN = 0.0; else if ((-d + b) > a) sN = sD;
            else { sN = (-d + b); sD = a; }
        }
        sc = (std::abs(sN) < 1e-7 ? 0.0 : sN / sD);
        tc = (std::abs(tN) < 1e-7 ? 0.0 : tN / tD);
        cv::Point2f dP = w + (sc * u) - (tc * v);
        return dist_sq(dP);
    }

    static ceres::CostFunction* Create(const vc::core::util::NormalGridVolume& normal_grid_volume, int plane_idx, double w_normal, double w_snap, bool direction_aware = false, int z_min = -1, int z_max = -1, bool invert_dir = false) {
        return new ceres::AutoDiffCostFunction<NormalConstraintPlane, 1, 3, 3, 3, 3>(
            new NormalConstraintPlane(normal_grid_volume, plane_idx, w_normal, w_snap, direction_aware, z_min, z_max, invert_dir)
        );
    }

    bool direction_aware_;
};


struct PointCorrectionLoss2P {
    PointCorrectionLoss2P(const cv::Vec3f& correction_src, const cv::Vec3f& correction_tgt, const cv::Vec2i& grid_loc_int)
        : correction_src_(correction_src), correction_tgt_(correction_tgt), grid_loc_int_(grid_loc_int) {}

    template <typename T>
    bool operator()(const T* const p00, const T* const p01, const T* const p10, const T* const p11, const T* const grid_loc, T* residual) const {
        // Calculate the local coordinates (u,v) within the quad by subtracting the integer grid location.
        T u = grid_loc[0] - T(grid_loc_int_[0]);
        T v = grid_loc[1] - T(grid_loc_int_[1]);

        // If the grid location is outside this specific quad (i.e., u,v not in [0,1]), this loss is zero.
        if (u < T(0.0) || u >= T(1.0) || v < T(0.0) || v >= T(1.0)) {
            residual[0] = T(0.0);
            residual[1] = T(0.0);
            return true;
        }

        // Bilinear interpolation to find the 3D point on the surface patch corresponding to the grid location.
        T p_interp[3];
        p_interp[0] = (T(1) - u) * (T(1) - v) * p00[0] + u * (T(1) - v) * p10[0] + (T(1) - u) * v * p01[0] + u * v * p11[0];
        p_interp[1] = (T(1) - u) * (T(1) - v) * p00[1] + u * (T(1) - v) * p10[1] + (T(1) - u) * v * p01[1] + u * v * p11[1];
        p_interp[2] = (T(1) - u) * (T(1) - v) * p00[2] + u * (T(1) - v) * p10[2] + (T(1) - u) * v * p01[2] + u * v * p11[2];

        // Residual 1: 3D Euclidean distance between the interpolated point and the target correction point.
        T dx_abs = p_interp[0] - T(correction_tgt_[0]);
        T dy_abs = p_interp[1] - T(correction_tgt_[1]);
        T dz_abs = p_interp[2] - T(correction_tgt_[2]);
        residual[0] = T(100)*ceres::sqrt(dx_abs * dx_abs + dy_abs * dy_abs + dz_abs * dz_abs);

        // Residual 2: 3D point-to-line distance from the interpolated point to the line defined by src -> tgt.
        T p1[3] = {T(correction_src_[0]), T(correction_src_[1]), T(correction_src_[2])};
        T p2[3] = {T(correction_tgt_[0]), T(correction_tgt_[1]), T(correction_tgt_[2])};

        T v_line[3] = {p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]};
        T w_line[3] = {p_interp[0] - p1[0], p_interp[1] - p1[1], p_interp[2] - p1[2]};

        T c1 = w_line[0] * v_line[0] + w_line[1] * v_line[1] + w_line[2] * v_line[2];
        T c2 = v_line[0] * v_line[0] + v_line[1] * v_line[1] + v_line[2] * v_line[2];

        T b = c1 / c2;
        T pb[3] = {p1[0] + b * v_line[0], p1[1] + b * v_line[1], p1[2] + b * v_line[2]};

        T dx_line = p_interp[0] - pb[0];
        T dy_line = p_interp[1] - pb[1];
        T dz_line = p_interp[2] - pb[2];

        residual[1] = T(100)*ceres::sqrt(dx_line * dx_line + dy_line * dy_line + dz_line * dz_line);

        return true;
    }

    static ceres::CostFunction* Create(const cv::Vec3f& correction_src, const cv::Vec3f& correction_tgt, const cv::Vec2i& grid_loc_int) {
        return new ceres::AutoDiffCostFunction<PointCorrectionLoss2P, 2, 3, 3, 3, 3, 2>(
            new PointCorrectionLoss2P(correction_src, correction_tgt, grid_loc_int)
        );
    }

private:
    const cv::Vec3f correction_src_;
    const cv::Vec3f correction_tgt_;
    const cv::Vec2i grid_loc_int_;
};


struct PointCorrectionLoss {
    PointCorrectionLoss(const cv::Vec3f& correction_src, const cv::Vec3f& correction_tgt, const cv::Vec2i& grid_loc_int)
    : correction_src_(correction_src), correction_tgt_(correction_tgt), grid_loc_int_(grid_loc_int) {}

    template <typename T>
    bool operator()(const T* const p00, const T* const p01, const T* const p10, const T* const p11, const T* const grid_loc, T* residual) const {
        // Calculate the local coordinates (u,v) within the quad by subtracting the integer grid location.
        T u = grid_loc[0] - T(grid_loc_int_[0]);
        T v = grid_loc[1] - T(grid_loc_int_[1]);

        // If the grid location is outside this specific quad (i.e., u,v not in [0,1]), this loss is zero.
        if (u < T(0.0) || u >= T(1.0) || v < T(0.0) || v >= T(1.0)) {
            residual[0] = T(0.0);
            return true;
        }

        // Bilinear interpolation to find the 3D point on the surface patch corresponding to the grid location.
        T p_interp[3];
        p_interp[0] = (T(1) - u) * (T(1) - v) * p00[0] + u * (T(1) - v) * p10[0] + (T(1) - u) * v * p01[0] + u * v * p11[0];
        p_interp[1] = (T(1) - u) * (T(1) - v) * p00[1] + u * (T(1) - v) * p10[1] + (T(1) - u) * v * p01[1] + u * v * p11[1];
        p_interp[2] = (T(1) - u) * (T(1) - v) * p00[2] + u * (T(1) - v) * p10[2] + (T(1) - u) * v * p01[2] + u * v * p11[2];

        // Residual 1: 3D Euclidean distance between the interpolated point and the target correction point.
        T dx_abs = p_interp[0] - T(correction_tgt_[0]);
        T dy_abs = p_interp[1] - T(correction_tgt_[1]);
        T dz_abs = p_interp[2] - T(correction_tgt_[2]);
        residual[0] = T(100)*ceres::sqrt(dx_abs * dx_abs + dy_abs * dy_abs + dz_abs * dz_abs);

        return true;
    }

    static ceres::CostFunction* Create(const cv::Vec3f& correction_src, const cv::Vec3f& correction_tgt, const cv::Vec2i& grid_loc_int) {
        return new ceres::AutoDiffCostFunction<PointCorrectionLoss, 1, 3, 3, 3, 3, 2>(
            new PointCorrectionLoss(correction_src, correction_tgt, grid_loc_int)
        );
    }

private:
    const cv::Vec3f correction_src_;
    const cv::Vec3f correction_tgt_;
    const cv::Vec2i grid_loc_int_;
};

struct PointsCorrectionLoss {
    PointsCorrectionLoss(std::vector<cv::Vec3f> tgts, std::vector<cv::Vec2f> grid_locs, cv::Vec2i grid_loc_int)
        : tgts_(std::move(tgts)), grid_locs_(std::move(grid_locs)), grid_loc_int_(grid_loc_int) {}

    template <typename T>
    bool operator()(T const* const* parameters, T* residuals) const {
        const T* p00 = parameters[0];
        const T* p01 = parameters[1];
        const T* p10 = parameters[2];
        const T* p11 = parameters[3];

        residuals[0] = T(0.0);
        for (size_t i = 0; i < tgts_.size(); ++i) {
            const T grid_loc[2] = {T(grid_locs_[i][0]), T(grid_locs_[i][1])};
            residuals[0] += T(0.1)*calculate_residual_for_point(i, p00, p01, p10, p11, grid_loc);
        }
        return true;
    }

private:
    template <typename T>
    T calculate_residual_for_point(int point_idx, const T* const p00, const T* const p01, const T* const p10, const T* const p11, const T* const grid_loc) const {
        const cv::Vec3f& tgt_cv = tgts_[point_idx];
        T tgt[3] = {T(tgt_cv[0]), T(tgt_cv[1]), T(tgt_cv[2])};

        // T u = grid_loc[0] - T(grid_loc_int_[0]);
        // T v = grid_loc[1] - T(grid_loc_int_[1]);
        //
        // if (u < T(0.0) || u > T(1.0) || v < T(0.0) || v > T(1.0)) {
        //     return T(0.0);
        // }

        T total_residual = T(0.0);

        // Non-differentiable 2D distance weight calculation with linear falloff
        double grid_loc_val[2] = {val(grid_loc[0]), val(grid_loc[1])};
        double dx_2d = grid_loc_val[0] - grid_loc_int_[0];
        double dy_2d = grid_loc_val[1] - grid_loc_int_[1];
        double dist_2d = std::sqrt(dx_2d * dx_2d + dy_2d * dy_2d);
        double weight_2d = std::max(0.0, 1.0 - dist_2d / 2.0);

        // Corner p00 (neighbors p10, p01)
        total_residual += calculate_corner_residual(tgt, p00, p10, p01);
        // Corner p10 (neighbors p00, p11)
        total_residual += calculate_corner_residual(tgt, p10, p00, p11);
        // Corner p01 (neighbors p00, p11)
        total_residual += calculate_corner_residual(tgt, p01, p00, p11);
        // Corner p11 (neighbors p10, p01)
        total_residual += calculate_corner_residual(tgt, p11, p10, p01);

        total_residual *= T(weight_2d);

        if (dbg_) {
            std::cout << "Point " << point_idx << " | Residual: " << val(total_residual) << std::endl;
        }
        return total_residual;
    }

    template <typename T>
    T calculate_corner_residual(const T* tgt, const T* p, const T* n1, const T* n2) const {
        // Vectors from p to its neighbors
        T v1[3] = { n1[0] - p[0], n1[1] - p[1], n1[2] - p[2] };
        T v2[3] = { n2[0] - p[0], n2[1] - p[1], n2[2] - p[2] };

        // Plane normal vector (cross product)
        T normal[3];
        normal[0] = v1[1] * v2[2] - v1[2] * v2[1];
        normal[1] = v1[2] * v2[0] - v1[0] * v2[2];
        normal[2] = v1[0] * v2[1] - v1[1] * v2[0];

        T norm_len = ceres::sqrt(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]);
        if (norm_len < T(1e-9)) return T(0.0);
        normal[0] /= norm_len;
        normal[1] /= norm_len;
        normal[2] /= norm_len;

        // Vector from a point on the plane (p) to the target point
        T w[3] = { tgt[0] - p[0], tgt[1] - p[1], tgt[2] - p[2] };

        // Differentiable (in the point) signed distance from tgt to the plane
        T dist = normal[0] * w[0] + normal[1] * w[1] + normal[2] * w[2];

        if (dbg_)
            std::cout << "dist " << dist << std::endl;

        // Non-differentiable weight calculation
        double tgt_val[3] = {val(tgt[0]), val(tgt[1]), val(tgt[2])};
        double p_val[3] = {val(p[0]), val(p[1]), val(p[2])};
        double normal_val[3] = {val(normal[0]), val(normal[1]), val(normal[2])};
        double w_val[3] = {tgt_val[0] - p_val[0], tgt_val[1] - p_val[1], tgt_val[2] - p_val[2]};
        double dist_val = normal_val[0] * w_val[0] + normal_val[1] * w_val[1] + normal_val[2] * w_val[2];

        double proj_val[3];
        proj_val[0] = tgt_val[0] - dist_val * normal_val[0];
        proj_val[1] = tgt_val[1] - dist_val * normal_val[1];
        proj_val[2] = tgt_val[2] - dist_val * normal_val[2];

        // Calculate the 3D distance between the projected point and the corner point p
        double dx_proj = proj_val[0] - p_val[0];
        double dy_proj = proj_val[1] - p_val[1];
        double dz_proj = proj_val[2] - p_val[2];
        double dist_proj_to_corner = std::sqrt(dx_proj*dx_proj + dy_proj*dy_proj + dz_proj*dz_proj);

        // Linear falloff from 1 to 0 over a distance of 40
        double weight = std::max(0.0, 1.0 - dist_proj_to_corner / 40.0);

        if (dbg_)
            std::cout << "weight " << weight << std::endl;
        return T(weight) * ceres::abs(dist);
    }

    std::vector<cv::Vec3f> tgts_;
    std::vector<cv::Vec2f> grid_locs_;
    const cv::Vec2i grid_loc_int_;
public:
    bool dbg_ = false;
};
