#include "LineStripInternal.hpp"
#include <Eigen/SparseCholesky>
#include <Eigen/SVD>
#include <limits>
#include <numeric>

namespace vc::lasagna::strip {
Grid fromMat(const cv::Mat_<cv::Vec3f>& m) {
    if(m.rows<3||m.rows%2!=1||m.cols<2)throw std::invalid_argument("Robust strip needs odd rows >=3 and at least two columns");
    Grid g(m.rows,m.cols);
    for(int r=0;r<m.rows;++r)for(int c=0;c<m.cols;++c) {
        g(r,c)=Vec(m(r,c));
        for(double v:g(r,c).val)if(!std::isfinite(v))throw std::invalid_argument("Non-finite strip coordinates");
    }
    return g;
}
cv::Mat_<cv::Vec3f> toMat(const Grid& g) {
    cv::Mat_<cv::Vec3f> m(g.rows,g.cols);
    for(int r=0;r<g.rows;++r)for(int c=0;c<g.cols;++c)m(r,c)=cv::Vec3f(g(r,c));
    return m;
}
Vec sample(const NormalSampler& sampler,Vec p,double fillRadius) {
    auto usable=[](const NormalSample& s){return s.valid&&std::isfinite(s.normal.dot(s.normal))&&cv::norm(s.normal)>1e-12;};
    const auto initial=sampler.sampleNormal(p);
    if(usable(initial))return unit(initial.normal);
    for(double f:{.125,.25,.5,1.}) {
        const double radius=fillRadius*f;Vec sum{0,0,0},reference{0,0,0};int count=0;
        for(int axis=0;axis<3;++axis)for(int sign:{-1,1}) {
            Vec q=p;q[axis]+=sign*radius;auto s=sampler.sampleNormal(q);
            if(!usable(s))continue;
            auto n=unit(s.normal);if(!count)reference=n;if(n.dot(reference)<0)n=-n;sum+=n;++count;
        }
        if(count>=2&&cv::norm(sum)/count>=.9)return unit(sum);
    }
    throw std::runtime_error("Robust strip: missing normal near ("+std::to_string(p[0])+", "+std::to_string(p[1])+", "+std::to_string(p[2])+"): "+initial.reason);
}
Grid normals(const Grid& g,const NormalSampler& sampler,const RobustLineStripOptions& o) {
    Grid n(g.rows,g.cols);
    for(int r=0;r<g.rows;++r) {
        check(o);
        for(int c=0;c<g.cols;++c)n(r,c)=sample(sampler,g(r,c),4*o.crossSpacing);
    }
    const Vec reference=n(g.rows/2,g.cols/2);
    for(auto& v:n.points)if(v.dot(reference)<0)v=-v;
    return n;
}
static bool badTriangle(Vec u,Vec v,double limit) {
    const double a=u.dot(u),b=u.dot(v),c=v.dot(v),lo=1/(limit*limit),hi=limit*limit;
    const double dl=(a-lo)*(c-lo)-b*b,dh=(hi-a)*(hi-c)-b*b;
    const double band=1e-10*std::max({1.,a*a,c*c,b*b});
    if(!std::isfinite(a+b+c))return true;
    if(std::min({std::abs(a-lo),std::abs(c-lo),std::abs(hi-a),std::abs(hi-c),std::abs(dl),std::abs(dh)})<=band) {
        Eigen::Matrix<double,3,2> j;for(int k=0;k<3;++k){j(k,0)=u[k];j(k,1)=v[k];}
        const auto s=Eigen::JacobiSVD<Eigen::Matrix<double,3,2>>(j).singularValues();
        return s[0]>limit||s[1]<1/limit;
    }
    return a<lo||c<lo||a>hi||c>hi||dl<0||dh<0;
}
std::vector<unsigned char> badQuads(const Grid& g,const RobustLineStripOptions& o,double limit) {
    std::vector<unsigned char> bad(size_t(g.rows-1)*(g.cols-1));
    for(int r=0;r<g.rows-1;++r)for(int c=0;c<g.cols-1;++c) {
        const auto a=g(r,c),b=g(r,c+1),d=g(r+1,c+1),e=g(r+1,c);
        const Vec u=(b-a)/o.alongSpacing,v=(e-a)/o.crossSpacing,
                  u2=(d-e)/o.alongSpacing,v2=(d-b)/o.crossSpacing;
        bad[size_t(r)*(g.cols-1)+c]=badTriangle(u,v,limit)||badTriangle(u2,v2,limit)||u.cross(v).dot(u2.cross(v2))<=0;
    }
    return bad;
}
bool valid(const Grid& g,const RobustLineStripOptions& o) {
    const auto b=badQuads(g,o);return std::none_of(b.begin(),b.end(),[](auto v){return v;});
}
std::vector<double> widths(const Grid& g) {
    std::vector<double> w(size_t(2)*g.cols);
    for(int r=0;r<g.rows-1;++r)for(int c=0;c<g.cols;++c)w[(r<g.rows/2?0:g.cols)+c]+=cv::norm(g(r+1,c)-g(r,c));
    return w;
}
bool widthOK(const Grid& g,const RobustLineStripOptions& o) {
    const double half=(g.rows/2)*o.crossSpacing;
    for(double w:widths(g))if(w<half*.95||w>half*1.05)return false;
    return true;
}
std::vector<unsigned char> badColumns(const Grid& g,const RobustLineStripOptions& o,double limit) {
    std::vector<unsigned char> result(g.cols);const auto bad=badQuads(g,o,limit);
    for(int r=0;r<g.rows-1;++r)for(int c=0;c<g.cols-1;++c)if(bad[size_t(r)*(g.cols-1)+c])result[c]=result[c+1]=1;
    return result;
}
std::vector<double> distances(const std::vector<unsigned char>& mask) {
    std::vector<double> d(mask.size(),std::numeric_limits<double>::infinity());
    for(size_t c=0;c<mask.size();++c){if(mask[c])d[c]=0;else if(c)d[c]=d[c-1]+1;}
    for(int c=int(mask.size())-2;c>=0;--c)d[c]=std::min(d[c],d[c+1]+1);
    return d;
}
Grid gaussianAlong(const Grid& g,double sigma,bool nearest) {
    const int radius=int(4*sigma+.5);std::vector<double> weights(2*radius+1);double total=0;
    for(int i=-radius;i<=radius;++i)total+=(weights[i+radius]=std::exp(-.5*i*i/(sigma*sigma)));
    Grid out(g.rows,g.cols);
    for(int r=0;r<g.rows;++r)for(int c=0;c<g.cols;++c)for(int k=-radius;k<=radius;++k) {
        int x=c+k;
        if(nearest)x=std::clamp(x,0,g.cols-1);
        else while(x<0||x>=g.cols){if(x<0)x=-x-1;else x=2*g.cols-1-x;}
        out(r,c)+=g(r,x)*(weights[k+radius]/total);
    }
    return out;
}
static std::vector<unsigned char> invalid(const Grid& g,const Grid& base,bool width,const RobustLineStripOptions& o) {
    auto cols=badColumns(g,o);
    for(int r=0;r<g.rows-1;++r)for(int c=0;c<g.cols-1;++c) {
        auto face=[&](const Grid& q,bool second){return second?(q(r+1,c)-q(r+1,c+1)).cross(q(r,c+1)-q(r+1,c+1)):(q(r,c+1)-q(r,c)).cross(q(r+1,c)-q(r,c));};
        if(face(g,false).dot(face(base,false))<=0||face(g,true).dot(face(base,true))<=0)cols[c]=cols[c+1]=1;
    }
    if(width){const auto w=widths(g);const double half=(g.rows/2)*o.crossSpacing;for(int c=0;c<g.cols;++c)if(w[c]<.95*half||w[c]>1.05*half||w[g.cols+c]<.95*half||w[g.cols+c]>1.05*half)cols[c]=1;}
    return cols;
}
static Grid guard(const Grid& base,const Grid& proposed,bool width,const RobustLineStripOptions& o,std::vector<double>& factor) {
    factor.assign(base.cols,1.);
    for(int iteration=0;iteration<24;++iteration) {
        check(o);Grid g=base;
        for(int r=0;r<g.rows;++r)for(int c=0;c<g.cols;++c)g(r,c)+=(proposed(r,c)-base(r,c))*factor[c];
        auto a=invalid(g,base,width,o),b=invalid(quantize(g),base,width,o);
        bool any=false;for(int c=0;c<g.cols;++c){a[c]|=b[c];any|=a[c];}
        if(!any)return g;
        const auto distance=distances(a);
        for(int c=0;c<g.cols;++c){factor[c]*=std::min(1.,.5+distance[c]/16.);if(factor[c]<std::ldexp(1.,-14))factor[c]=0;}
    }
    factor.assign(base.cols,0.);return base;
}
static Grid robustFit(const Grid& base,const NormalSampler& sampler,bool width,const RobustLineStripOptions& o) {
    const Grid reference=normals(base,sampler,o);Grid current=reference,g=base;
    const int n=int(base.points.size()),middle=base.rows/2;std::vector<int> free(n,-1);int count=0;
    for(int r=0;r<base.rows;++r)for(int c=0;c<base.cols;++c)if(r!=middle)free[r*base.cols+c]=count++;
    std::vector<double> previous(n,0.);
    using Triplet=Eigen::Triplet<double>;using Matrix=Eigen::SparseMatrix<double>;
    for(int iteration=0;iteration<4;++iteration) {
        check(o);std::vector<Triplet> entries;entries.reserve(size_t(n)*30);Eigen::VectorXd rhs=Eigen::VectorXd::Zero(count);
        auto equation=[&](const std::vector<int>& indices,const std::vector<double>& coefficients,double target,double weight) {
            for(size_t a=0;a<indices.size();++a)if(free[indices[a]]>=0) {
                const int i=free[indices[a]];rhs[i]+=weight*coefficients[a]*target;
                for(size_t b=0;b<indices.size();++b)if(free[indices[b]]>=0)entries.emplace_back(i,free[indices[b]],weight*coefficients[a]*coefficients[b]);
            }
        };
        auto edge=[&](int i,int j) {
            const Vec normal=unit(current.points[i]+current.points[j]);
            const double a=-normal.dot(reference.points[i]),b=normal.dot(reference.points[j]),target=-normal.dot(base.points[j]-base.points[i]);
            const double residual=a*previous[i]+b*previous[j]-target;
            const double weight=iteration?std::min(1.,(.125*o.crossSpacing)/std::max(std::abs(residual),1e-12)):1.;
            equation({i,j},{a,b},target,weight);
        };
        for(int r=0;r<base.rows;++r)for(int c=0;c<base.cols-1;++c)edge(r*base.cols+c,r*base.cols+c+1);
        for(int r=0;r<base.rows-1;++r)for(int c=0;c<base.cols;++c)edge(r*base.cols+c,(r+1)*base.cols+c);
        for(int r=0;r<base.rows;++r)for(int c=0;c<base.cols-2;++c)equation({r*base.cols+c,r*base.cols+c+1,r*base.cols+c+2},{1,-2,1},0,.25);
        for(int r=0;r<base.rows-2;++r)for(int c=0;c<base.cols;++c)equation({r*base.cols+c,(r+1)*base.cols+c,(r+2)*base.cols+c},{1,-2,1},0,.25);
        for(int i=0;i<count;++i)entries.emplace_back(i,i,.0025);
        Matrix hessian(count,count);hessian.setFromTriplets(entries.begin(),entries.end());
        Eigen::SimplicialLDLT<Matrix> solver;solver.compute(hessian);
        if(solver.info()!=Eigen::Success)throw std::runtime_error("Robust strip normal-fit factorization failed");
        const Eigen::VectorXd h=solver.solve(rhs);
        if(solver.info()!=Eigen::Success||!h.allFinite())throw std::runtime_error("Robust strip normal-fit solve failed");
        Grid proposed=base;std::vector<double> height(n,0.);
        for(int i=0;i<n;++i)if(free[i]>=0){height[i]=std::clamp(h[free[i]],-3*o.crossSpacing,3*o.crossSpacing);proposed.points[i]+=height[i]*reference.points[i];}
        std::vector<double> factor;g=guard(base,proposed,width,o,factor);
        for(int i=0;i<n;++i)previous[i]=height[i]*factor[i%base.cols];
        if(iteration<3)current=normals(g,sampler,o);
    }
    return quantize(g);
}
static Grid follow(const Grid& ribbon,const NormalSampler& sampler,int span,const RobustLineStripOptions& o) {
    Grid g(ribbon.rows,ribbon.cols);const int middle=g.rows/2;
    const int chord=std::max(1,int(std::lround(span/o.alongSpacing/2)));
    for(int c=0;c<g.cols;++c)g(middle,c)=ribbon(middle,c);
    for(int sign:{-1,1})for(int c=0;c<g.cols;++c) {
        if(c%128==0)check(o);
        const Vec tangent=unit(ribbon(middle,std::min(g.cols-1,c+chord))-ribbon(middle,std::max(0,c-chord)));
        Vec p=ribbon(middle,c),previous=unit(ribbon(middle+1,c)-p)*sign;
        auto side=[&](Vec at,Vec prev){Vec s=unit(tangent.cross(sample(sampler,at,4*o.crossSpacing)));if(cv::norm(s)<.5)throw std::runtime_error("Robust strip tangent is parallel to sheet normal");return s.dot(prev)<0?-s:s;};
        for(int r=1;r<=middle;++r){const Vec k1=side(p,previous),k2=side(p+.5*o.crossSpacing*k1,k1);p+=o.crossSpacing*k2;previous=k2;g(middle+sign*r,c)=p;}
    }
    return quantize(g);
}
static void validateOptions(const RobustLineStripOptions& o) {
    if(!std::isfinite(o.alongSpacing)||!std::isfinite(o.crossSpacing)||o.alongSpacing<=0||o.crossSpacing<=0)throw std::invalid_argument("Invalid robust strip spacing");
}
} // namespace vc::lasagna::strip

namespace vc::lasagna {
cv::Mat_<cv::Vec3f> fitRobustLineStrip(const cv::Mat_<cv::Vec3f>& seed,const NormalSampler& sampler,bool width,const RobustLineStripOptions& o) {
    strip::validateOptions(o);auto base=strip::fromMat(seed);
    if(!strip::valid(base,o)||(width&&!strip::widthOK(base,o)))throw std::runtime_error("Invalid robust strip seed");
    return strip::toMat(strip::robustFit(base,sampler,width,o));
}
cv::Mat_<cv::Vec3f> buildRobustLineStrip(const cv::Mat_<cv::Vec3f>& ribbon,const NormalSampler& sampler,const RobustLineStripOptions& o) {
    strip::validateOptions(o);const auto initial=strip::fromMat(ribbon);auto base=initial;size_t fewest=std::numeric_limits<size_t>::max();bool repaired=false;
    for(int span:{40,80,120}) {
        auto candidate=strip::follow(initial,sampler,span,o);const auto bad=strip::badQuads(candidate,o);const size_t count=std::count(bad.begin(),bad.end(),1);
        if(count<fewest){base=std::move(candidate);fewest=count;}
        if(!count)break;
    }
    if(fewest){base=strip::repairSeed(base,strip::normals(base,sampler,o),o);repaired=true;}
    if(!strip::valid(base,o)||(repaired&&!strip::widthOK(base,o)))throw std::runtime_error("No geometry-valid full-width robust strip repair");
    return strip::toMat(strip::robustFit(base,sampler,repaired,o));
}
} // namespace vc::lasagna
