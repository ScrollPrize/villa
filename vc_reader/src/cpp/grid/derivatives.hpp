#pragma once
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
    VCR_DERIV_HD bool ok(int r, int c) const { return r>=0 && c>=0 && r<rows && c<cols && v.valid[r*cols+c]; }
    VCR_DERIV_HD bool same(int c, int d) const { return !v.piece || v.piece[c]==v.piece[d]; }
    VCR_DERIV_HD geometry::V get(const float* p,int r,int c) const {return geometry::vector(p+3*(r*cols+c));}
    VCR_DERIV_HD geometry::V first(const float* p,int r,int c,bool column) const {
        int dr=column?0:1, dc=column?1:0;
        bool lo=ok(r-dr,c-dc)&&(!column||same(c-dc,c));
        bool hi=ok(r+dr,c+dc)&&(!column||same(c,c+dc));
        if(lo&&hi)return (get(p,r+dr,c+dc)-get(p,r-dr,c-dc))*.5f;
        // One sided: second order when the next vertex on that side is valid too; a first
        // order difference leaves border and hole edge cells with bilinear-grade normals.
        if(hi){
            if(ok(r+2*dr,c+2*dc)&&(!column||same(c+dc,c+2*dc)))
                return (get(p,r+dr,c+dc)*4.f-get(p,r,c)*3.f-get(p,r+2*dr,c+2*dc))*.5f;
            return get(p,r+dr,c+dc)-get(p,r,c);
        }
        if(lo){
            if(ok(r-2*dr,c-2*dc)&&(!column||same(c-2*dc,c-dc)))
                return (get(p,r,c)*3.f-get(p,r-dr,c-dc)*4.f+get(p,r-2*dr,c-2*dc))*.5f;
            return get(p,r,c)-get(p,r-dr,c-dc);
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
