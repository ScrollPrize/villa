#pragma once

#include "vc/core/util/SymmetricDirichlet.hpp"
#include <ceres/ceres.h>
#include <algorithm>
#include <limits>
#include <vector>

namespace vc::constraints::detail {
struct ChartMetric {
    double inv00,inv01,inv11,weight,orientation;
    template<class T> bool operator()(const T* a,const T* b,const T* c,T* r) const {
        const T j00=(b[0]-a[0])*T(inv00), j10=(b[1]-a[1])*T(inv00);
        const T j01=(b[0]-a[0])*T(inv01)+(c[0]-a[0])*T(inv11);
        const T j11=(b[1]-a[1])*T(inv01)+(c[1]-a[1])*T(inv11);
        const T determinant=j00*j11-j01*j10;
        // Refuse flipped/degenerate trial steps, rather than regularizing an
        // inverse through zero and silently accepting a reflected element.
        if(dirichletValue(determinant)*orientation<=1e-8) return false;
        const T w=T(weight);
        r[0]=w*j00;r[1]=w*j01;r[2]=w*j10;r[3]=w*j11;
        r[4]=w*j11/determinant;r[5]=-w*j01/determinant;
        r[6]=-w*j10/determinant;r[7]=w*j00/determinant;
        return true;
    }
};

// Evaluate the same residual arithmetic as ChartMetric<double>. Its Jacobian
// is a constant edge-to-J map followed by the derivative of J and inverse(J).
// Avoid constructing six-component Jets for every triangle on every iteration.
class ChartMetricCost final : public ceres::SizedCostFunction<8,2,2,2> {
    ChartMetric metric_;
public:
    explicit ChartMetricCost(ChartMetric metric):metric_(metric) {}
    bool Evaluate(double const* const* p,double* r,double** jacobians)const override {
        if(!metric_(p[0],p[1],p[2],r))return false;
        if(!jacobians)return true;
        const auto& m=metric_;
        const double a=(p[1][0]-p[0][0])*m.inv00;
        const double b=(p[1][0]-p[0][0])*m.inv01+(p[2][0]-p[0][0])*m.inv11;
        const double c=(p[1][1]-p[0][1])*m.inv00;
        const double d=(p[1][1]-p[0][1])*m.inv01+(p[2][1]-p[0][1])*m.inv11;
        const double det=a*d-b*c;
        const double du[3]={-m.inv00,m.inv00,0};
        const double dv[3]={-m.inv01-m.inv11,m.inv01,m.inv11};
        for(int i=0;i<3;++i)if(jacobians[i])for(int axis=0;axis<2;++axis) {
            const double da=axis==0?du[i]:0,db=axis==0?dv[i]:0;
            const double dc=axis==1?du[i]:0,dd=axis==1?dv[i]:0;
            const double deltaDet=da*d+a*dd-db*c-b*dc;
            const double numerator[4]={d,-b,-c,a};
            const double delta[4]={dd,-db,-dc,da};
            auto* j=jacobians[i];
            j[axis]=m.weight*da;j[2+axis]=m.weight*db;
            j[4+axis]=m.weight*dc;j[6+axis]=m.weight*dd;
            for(int k=0;k<4;++k)
                j[2*(k+4)+axis]=m.weight*(delta[k]/det-numerator[k]*deltaDet/(det*det));
        }
        return true;
    }
};

// Keep the ordinary symmetric Dirichlet energy, and penalize its excess per
// triangle. The additional term is flat at an isometry but grows quartically
// with stretch: a few damaged elements cannot cheaply absorb an overlap's
// entire displacement. Reuse the same no-flip barrier and analytic Jacobian.
class ReviewChartMetricCost final : public ceres::SizedCostFunction<9,2,2,2> {
    double weight_;
    ChartMetricCost metric_;
public:
    explicit ReviewChartMetricCost(ChartMetric metric)
        :weight_(metric.weight),metric_([&]{metric.weight=1;return metric;}()) {}
    double distortion(double const* const* p)const {
        double r[8];
        if(!metric_.Evaluate(p,r,nullptr))return std::numeric_limits<double>::infinity();
        // Singular values of J: measure the worst stretch OR compression,
        // including shear that an edge-length percentile can miss.
        const double a=r[0]*r[0]+r[2]*r[2],b=r[1]*r[1]+r[3]*r[3],c=r[0]*r[1]+r[2]*r[3];
        const double largest=std::sqrt(.5*(a+b+std::hypot(a-b,2*c)));
        const double determinant=std::abs(r[0]*r[3]-r[1]*r[2]);
        return std::max(largest,largest/determinant);
    }
    bool Evaluate(double const* const* p,double* r,double** jacobians)const override {
        double values[8],derivatives[3][16];
        double* blocks[3]={nullptr,nullptr,nullptr};
        if(jacobians)for(int i=0;i<3;++i)if(jacobians[i])blocks[i]=derivatives[i];
        if(!metric_.Evaluate(p,values,jacobians?blocks:nullptr))return false;
        double energy=0;
        for(int k=0;k<8;++k){energy+=values[k]*values[k];r[k]=weight_*values[k];}
        r[8]=weight_*.25*std::max(0.,energy-4.);
        if(jacobians)for(int i=0;i<3;++i)if(jacobians[i])for(int axis=0;axis<2;++axis) {
            double derivative=0;
            for(int k=0;k<8;++k){
                jacobians[i][2*k+axis]=weight_*derivatives[i][2*k+axis];
                derivative+=values[k]*derivatives[i][2*k+axis];
            }
            jacobians[i][16+axis]=energy>4?weight_*.5*derivative:0;
        }
        return true;
    }
};

// Correspondences are linear. Dynamic autodiff otherwise evaluates each one
// repeatedly in strides of four derivatives, even though its Jacobian is fixed.
class MatchCost final : public ceres::CostFunction {
    std::vector<double> coefficients_;
    double weight_;
public:
    MatchCost(std::vector<double> coefficients,double weight)
        :coefficients_(std::move(coefficients)),weight_(weight) {
        set_num_residuals(2);
        mutable_parameter_block_sizes()->assign(coefficients_.size(),2);
    }
    bool Evaluate(double const* const* p,double* r,double** jacobians)const override {
        r[0]=0;r[1]=0;
        for(size_t i=0;i<coefficients_.size();++i) {
            const double c=coefficients_[i]*weight_;
            for(int d=0;d<2;++d)r[d]+=p[i][d]*c;
            if(jacobians&&jacobians[i]) {
                auto* j=jacobians[i];j[0]=c;j[1]=0;j[2]=0;j[3]=c;
            }
        }
        return true;
    }
};
} // namespace vc::constraints::detail
