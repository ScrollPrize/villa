#pragma once
#include <cmath>
#include "points.hpp"
#ifdef __CUDACC__
#define VCR_DERIV_HD __host__ __device__
#else
#define VCR_DERIV_HD
#endif
namespace vcr::grid {
// Hermite derivatives shared by stored surfaces and resident CPU/CUDA grids.
struct Derivatives {
    GridView v;
    int rows, cols;
    // False: only the first order steps of the one sided stencils (the grid is a window whose context is cut).
    bool secondOrder = true;
    VCR_DERIV_HD bool ok(int r, int c) const { return r>=0 && c>=0 && r<rows && c<cols && v.valid[r*cols+c]; }
    VCR_DERIV_HD bool same(int c, int d) const { return !v.piece || v.piece[c]==v.piece[d]; }
    VCR_DERIV_HD geometry::V get(const float* p,int r,int c) const {return geometry::vector(p+3*(r*cols+c));}
    // Second order tangent at a, looking toward b then c: (3 s1 - s2) / 2 with steps s1=b-a,
    // s2=c-b. It is used only when s2 differs from s1 by at most |s1|/2, which keeps it within
    // 15 degrees of the secant s1 (never reversed); otherwise the secant.
    VCR_DERIV_HD static geometry::V edge(geometry::V a,geometry::V b,geometry::V c){
        geometry::V s=b-a,e=s-(c-b);
        return geometry::dot(e,e)<=geometry::dot(s,s)*.25f?s+e*.5f:s;
    }
    // The tangent main takes at a vertex: centred, or the first order step toward the one
    // valid side; zero when neither side is valid.
    VCR_DERIV_HD geometry::V plain(const float* p,int r,int c,bool column) const {
        int dr=column?0:1, dc=column?1:0;
        bool lo=ok(r-dr,c-dc)&&(!column||same(c-dc,c));
        bool hi=ok(r+dr,c+dc)&&(!column||same(c,c+dc));
        if(lo&&hi)return (get(p,r+dr,c+dc)-get(p,r-dr,c-dc))*.5f;
        if(hi)return get(p,r+dr,c+dc)-get(p,r,c);
        if(lo)return get(p,r,c)-get(p,r-dr,c-dc);
        return {0,0,0};
    }
    // The second order tangent t of a vertex's positions replaces the secant s only where it stays
    // within about 6 degrees of it and keeps the cross product with the other axis' tangent w on the
    // secant's side, above half its size: a small turn of a tangent can still flip the patch normal
    // on a skewed or folded cell, and a flipped normal sends a signed layer to the other side of
    // the sheet. On a smooth sheet the correction is a fraction of the step's turn, far inside both.
    VCR_DERIV_HD geometry::V oriented(geometry::V t,geometry::V s,const float* p,int r,int c,bool column) const {
        if(p!=v.xyz)return t;
        if(geometry::dot(t,s)<.995f*sqrtf(geometry::dot(t,t)*geometry::dot(s,s)))return s;
        geometry::V w=plain(v.xyz,r,c,!column),cs=geometry::cross(s,w),ct=geometry::cross(t,w);
        return geometry::dot(ct,cs)>=geometry::dot(cs,cs)*.5f?t:s;
    }
    VCR_DERIV_HD geometry::V first(const float* p,int r,int c,bool column) const {
        int dr=column?0:1, dc=column?1:0;
        bool lo=ok(r-dr,c-dc)&&(!column||same(c-dc,c));
        bool hi=ok(r+dr,c+dc)&&(!column||same(c,c+dc));
        if(lo&&hi)return (get(p,r+dr,c+dc)-get(p,r-dr,c-dc))*.5f;
        // One sided: second order when the next vertex on that side is valid too.
        if(hi){
            geometry::V a=get(p,r,c),b=get(p,r+dr,c+dc);
            if(secondOrder&&ok(r+2*dr,c+2*dc)&&(!column||same(c+dc,c+2*dc)))
                return oriented(edge(a,b,get(p,r+2*dr,c+2*dc)),b-a,p,r,c,column);
            return b-a;
        }
        if(lo){
            geometry::V a=get(p,r,c),b=get(p,r-dr,c-dc);
            if(secondOrder&&ok(r-2*dr,c-2*dc)&&(!column||same(c-2*dc,c-dc)))
                return oriented(edge(a,b,get(p,r-2*dr,c-2*dc)),b-a,p,r,c,column)*-1.f;
            return a-b;
        }
        return {0,0,0};
    }
    VCR_DERIV_HD void firstAt(int at,float* du,float* dv) const {
        int r=at/cols,c=at%cols;
        geometry::store(du+3*at,ok(r,c)?first(v.xyz,r,c,true):geometry::V{0,0,0});
        geometry::store(dv+3*at,ok(r,c)?first(v.xyz,r,c,false):geometry::V{0,0,0});
    }
    VCR_DERIV_HD void mixedAt(int at,float* out) const {
        int r=at/cols,c=at%cols; geometry::V value{0,0,0};
        if(ok(r,c)) {
            bool left=ok(r,c-1)&&same(c-1,c),right=ok(r,c+1)&&same(c,c+1);
            bool up=ok(r-1,c),down=ok(r+1,c);
            if(left&&right&&up&&down&&ok(r-1,c-1)&&ok(r-1,c+1)&&ok(r+1,c-1)&&ok(r+1,c+1))
                value=(get(v.xyz,r+1,c+1)-get(v.xyz,r+1,c-1)-get(v.xyz,r-1,c+1)+get(v.xyz,r-1,c-1))*.25f;
            else {
                int n=int(up||down)+int(left||right);
                if(n)value=(first(v.du,r,c,false)+first(v.dv,r,c,true))*(1.f/n);
            }
        }
        geometry::store(out+3*at,value);
    }
};
}
#undef VCR_DERIV_HD
