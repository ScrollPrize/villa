#pragma once

#include "vc/core/types/ChunkedTensor.hpp"

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

struct StraightLoss {
    StraightLoss(float w) : _w(w) {};
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
        
        T dot = (d1[0]*d2[0] + d1[1]*d2[1] + d1[2]*d2[2])/(l1*l2);
        
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
    FiberDirectionLoss(Chunked3dVec3fFromUint8 &fiber_dirs, float w) : _fiber_dirs(fiber_dirs), _w(w) {};
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

        E const patch_u_dist = sqrt(patch_u_disp_zyx[0] * patch_u_disp_zyx[0] + patch_u_disp_zyx[1] * patch_u_disp_zyx[1] + patch_u_disp_zyx[2] * patch_u_disp_zyx[2]);
        E const abs_dot = abs(patch_u_disp_zyx[0] * fiber_dir_zyx[0] + patch_u_disp_zyx[1] * fiber_dir_zyx[1] + patch_u_disp_zyx[2] * fiber_dir_zyx[2]) / patch_u_dist;

        residual[0] = E(_w) * (E(1) - abs_dot);

        return true;
    }

    static double unjet(const double& v) { return v; }
    template<typename JetT> static double unjet(const JetT& v) { return v.a; }

    float _w;
    Chunked3dVec3fFromUint8 &_fiber_dirs;

    static ceres::CostFunction* Create(Chunked3dVec3fFromUint8 &fiber_dirs, float w = 1.0)
    {
        return new ceres::AutoDiffCostFunction<FiberDirectionLoss, 1, 3, 3>(new FiberDirectionLoss(fiber_dirs, w));
    }
};

struct NormalDirectionLoss {
    NormalDirectionLoss(Chunked3dVec3fFromUint8 &normal_dirs, float w) : _normal_dirs(normal_dirs), _w(w) {};
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

        E const abs_dot = abs(patch_normal_zyx[0] * target_normal_zyx[0] + patch_normal_zyx[1] * target_normal_zyx[1] + patch_normal_zyx[2] * target_normal_zyx[2]) / patch_normal_length;

        residual[0] = E(_w) * (E(1) - abs_dot);

        return true;
    }

    static double unjet(const double& v) { return v; }
    template<typename JetT> static double unjet(const JetT& v) { return v.a; }

    float _w;
    Chunked3dVec3fFromUint8 &_normal_dirs;

    static ceres::CostFunction* Create(Chunked3dVec3fFromUint8 &normal_dirs, float w = 1.0)
    {
        return new ceres::AutoDiffCostFunction<NormalDirectionLoss, 1, 3, 3, 3>(new NormalDirectionLoss(normal_dirs, w));
    }
};
