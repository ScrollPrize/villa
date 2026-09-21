#pragma once
#include "vc/lasagna/RobustLineStrip.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace vc::lasagna::strip {
using Vec = cv::Vec3d;
struct Grid {
    int rows, cols;
    std::vector<Vec> points;
    Grid(int r, int c) : rows(r), cols(c), points(size_t(r)*c) {}
    Vec& operator()(int r, int c) { return points[size_t(r)*cols+c]; }
    const Vec& operator()(int r, int c) const { return points[size_t(r)*cols+c]; }
};
inline Vec unit(Vec v) { return v/std::max(cv::norm(v),1e-12); }
inline void check(const RobustLineStripOptions& o) { if(o.checkCancelled)o.checkCancelled(); }
inline Grid quantize(Grid g) {
    for(auto& p:g.points) for(int d=0;d<3;++d)p[d]=float(p[d]);
    return g;
}
Grid fromMat(const cv::Mat_<cv::Vec3f>& g);
cv::Mat_<cv::Vec3f> toMat(const Grid& g);
Grid normals(const Grid& g,const NormalSampler&,const RobustLineStripOptions&);
Vec sample(const NormalSampler&,Vec,double fillRadius);
std::vector<unsigned char> badQuads(const Grid&,const RobustLineStripOptions&,double limit=3.0);
bool valid(const Grid&,const RobustLineStripOptions&);
std::vector<double> widths(const Grid&);
bool widthOK(const Grid&,const RobustLineStripOptions&);
std::vector<unsigned char> badColumns(const Grid&,const RobustLineStripOptions&,double limit=3.0);
std::vector<double> distances(const std::vector<unsigned char>&);
Grid gaussianAlong(const Grid&,double sigma,bool nearest=false);
Grid repairSeed(const Grid&,const Grid& normals,const RobustLineStripOptions&);
} // namespace vc::lasagna::strip
