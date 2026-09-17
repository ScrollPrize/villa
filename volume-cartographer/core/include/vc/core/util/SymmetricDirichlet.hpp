#pragma once
#include <ceres/ceres.h>
#include <opencv2/core/types.hpp>
#include <cmath>

namespace vc {
inline double dirichletValue(double v) { return v; }
template<class T> double dirichletValue(const T& v) { return v.a; }
struct SymmetricDirichletLoss {
    SymmetricDirichletLoss(double unit, double w, double eps_abs, double eps_rel)
      : _unit(unit), _du0(0.0), _du1(1.0), _dv0(1.0), _dv1(0.0),
        _w(w), _eps_abs(eps_abs), _eps_rel(eps_rel) {}

    SymmetricDirichletLoss(double unit, double du0, double du1, double dv0, double dv1,
                           double w, double eps_abs, double eps_rel)
      : _unit(unit), _du0(du0), _du1(du1), _dv0(dv0), _dv1(dv1),
        _w(w), _eps_abs(eps_abs), _eps_rel(eps_rel) {}

    template <typename T>
    bool operator()(const T* const p,    // 3D at (i,j)
                    const T* const pu,   // 3D at (i,j+1)  (u-direction)
                    const T* const pv,   // 3D at (i+1,j)  (v-direction)
                    T* residual) const
    {
        for (int i = 0; i < 3; ++i) {
            if (!std::isfinite(dirichletValue(p[i])) || !std::isfinite(dirichletValue(pu[i])) || !std::isfinite(dirichletValue(pv[i])) ||
                dirichletValue(p[i]) == -1.0 || dirichletValue(pu[i]) == -1.0 || dirichletValue(pv[i]) == -1.0) {
                residual[0] = T(0);
                return true;
            }
        }

        // Normalize edge vectors by expected step length to get a dimensionless Jacobian
        const T inv_unit = T(1.0) / T(_unit);
        T eu[3] = { (pu[0]-p[0])*inv_unit, (pu[1]-p[1])*inv_unit, (pu[2]-p[2])*inv_unit };
        T ev[3] = { (pv[0]-p[0])*inv_unit, (pv[1]-p[1])*inv_unit, (pv[2]-p[2])*inv_unit };

        // First fundamental form G = J^T J is 2x2 SPD with entries a,b,c
        T a = eu[0]*eu[0] + eu[1]*eu[1] + eu[2]*eu[2];
        T c = ev[0]*ev[0] + ev[1]*ev[1] + ev[2]*ev[2];
        T b = eu[0]*ev[0] + eu[1]*ev[1] + eu[2]*ev[2];
        if (!std::isfinite(dirichletValue(a)) || !std::isfinite(dirichletValue(b)) || !std::isfinite(dirichletValue(c))) {
            residual[0] = T(0);
            return true;
        }

        const T ea = T(_du0 * _du0 + _du1 * _du1);
        const T ec = T(_dv0 * _dv0 + _dv1 * _dv1);
        const T eb = T(_du0 * _dv0 + _du1 * _dv1);
        const T e_det = ea * ec - eb * eb;
        if (dirichletValue(e_det) <= 1e-12) {
            residual[0] = T(0);
            return true;
        }

        const T m00 = (ec * a - eb * b) / e_det;
        const T m11 = (-eb * b + ea * c) / e_det;
        const T m01 = (ec * b - eb * c) / e_det;
        const T m10 = (-eb * a + ea * b) / e_det;

        T trG  = m00 + m11;
        T detG = m00 * m11 - m01 * m10;
        T det_safe = detG + T(_eps_abs) + T(_eps_rel) * ceres::abs(trG);

        // E_SD(J) = tr(G) + tr(G^{-1}) ; for 2x2, tr(G^{-1}) = (a+c)/det(G)
        T E = trG + trG / det_safe;
        if (!std::isfinite(dirichletValue(E))) {
            residual[0] = T(0);
            return true;
        }
        // Zero at optimum (identity metric): E - 4
        residual[0] = T(_w) * (E - T(4.0));
        return true;
    }

    static ceres::CostFunction* Create(double unit, double w = 1.0, double eps_abs = 1e-8, double eps_rel = 1e-2)
    {
        return new ceres::AutoDiffCostFunction<SymmetricDirichletLoss, 1, 3, 3, 3>(
            new SymmetricDirichletLoss(unit, w, eps_abs, eps_rel));
    }

    static ceres::CostFunction* Create(double unit,
                                       const cv::Vec2i& du,
                                       const cv::Vec2i& dv,
                                       double w = 1.0,
                                       double eps_abs = 1e-8,
                                       double eps_rel = 1e-2)
    {
        return new ceres::AutoDiffCostFunction<SymmetricDirichletLoss, 1, 3, 3, 3>(
            new SymmetricDirichletLoss(unit,
                                       static_cast<double>(du[0]),
                                       static_cast<double>(du[1]),
                                       static_cast<double>(dv[0]),
                                       static_cast<double>(dv[1]),
                                       w, eps_abs, eps_rel));
    }

    double _unit;
    double _du0;
    double _du1;
    double _dv0;
    double _dv1;
    double _w;
    double _eps_abs;
    double _eps_rel;
};
}
