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
    int firstRow = 0, lastRow = rows, firstCol = 0, lastCol = cols;
    VCR_DERIV_HD bool ok(int r, int c) const {
        return r >= firstRow && c >= firstCol && r < lastRow && c < lastCol && v.valid[r*cols+c];
    }
    VCR_DERIV_HD bool same(int c, int d) const { return !v.piece || v.piece[c]==v.piece[d]; }
    VCR_DERIV_HD geometry::V get(const float* p,int r,int c) const {return geometry::vector(p+3*(r*cols+c));}
    template<class Value>
    VCR_DERIV_HD geometry::V difference(Value value, int r, int c, bool column) const {
        int dr=column?0:1, dc=column?1:0;
        bool lo=ok(r-dr,c-dc)&&(!column||same(c-dc,c));
        bool hi=ok(r+dr,c+dc)&&(!column||same(c,c+dc));
        if(lo&&hi)return (value(r+dr,c+dc)-value(r-dr,c-dc))*.5f;
        if(hi)return value(r+dr,c+dc)-value(r,c);
        if(lo)return value(r,c)-value(r-dr,c-dc);
        return {0,0,0};
    }
    VCR_DERIV_HD geometry::V first(int r, int c, bool column) const {
        return difference([&](int y, int x) { return get(v.xyz, y, x); }, r, c, column);
    }
    VCR_DERIV_HD void firstAt(int at,float* du,float* dv) const {
        int r=at/cols,c=at%cols;
        geometry::store(du+3*at,ok(r,c)?first(r,c,true):geometry::V{0,0,0});
        geometry::store(dv+3*at,ok(r,c)?first(r,c,false):geometry::V{0,0,0});
    }
    template<bool Stored>
    VCR_DERIV_HD geometry::V mixed(int r, int c) const {
        if(ok(r,c)) {
            bool left=ok(r,c-1)&&same(c-1,c),right=ok(r,c+1)&&same(c,c+1);
            bool up=ok(r-1,c),down=ok(r+1,c);
            if(left&&right&&up&&down&&ok(r-1,c-1)&&ok(r-1,c+1)&&ok(r+1,c-1)&&ok(r+1,c+1))
                return (get(v.xyz,r+1,c+1)-get(v.xyz,r+1,c-1)-get(v.xyz,r-1,c+1)+get(v.xyz,r-1,c-1))*.25f;
            else {
                int n=int(up||down)+int(left||right);
                if (n) {
                    const auto du = [&](int y, int x) {
                        if constexpr (Stored) return get(v.du, y, x);
                        else return first(y, x, true);
                    };
                    const auto dv = [&](int y, int x) {
                        if constexpr (Stored) return get(v.dv, y, x);
                        else return first(y, x, false);
                    };
                    return (difference(du, r, c, false) + difference(dv, r, c, true)) * (1.f / n);
                }
            }
        }
        return {0,0,0};
    }
    VCR_DERIV_HD void mixedAt(int at,float* out) const {
        geometry::store(out+3*at,mixed<true>(at/cols,at%cols));
    }
};
}
#undef VCR_DERIV_HD
