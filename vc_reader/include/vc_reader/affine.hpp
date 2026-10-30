#pragma once
#include <array>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <optional>
#include <span>
#include <stdexcept>
#include <nlohmann/json.hpp>

namespace vcr::catalog {
// XYZ column vectors. Double coefficients preserve catalog precision and the
// float64 coordinates used by point collections, fibers and umbilici.
struct Affine {
    std::array<double, 16> values{1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    static Affine identity() { return {}; }
    double& operator()(int r, int c) { return values.at(r*4+c); }
    double operator()(int r, int c) const { return values.at(r*4+c); }
    static Affine scale(double factor) {
        if (!std::isfinite(factor) || factor <= 0) throw std::invalid_argument("scale must be finite and positive");
        Affine a; a(0,0)=a(1,1)=a(2,2)=factor; return a;
    }
    std::array<double,3> apply(const std::array<double,3>& p, bool vector=false) const {
        std::array<double,3> q{};
        for (int i=0;i<3;++i) q[i]=(*this)(i,0)*p[0]+(*this)(i,1)*p[1]+(*this)(i,2)*p[2]+(vector?0:(*this)(i,3));
        return q;
    }
    Affine operator*(const Affine& b) const {
        Affine c;
        for(int i=0;i<3;++i) for(int j=0;j<4;++j) {
            c(i,j)=j==3?(*this)(i,3):0;
            for(int k=0;k<3;++k)c(i,j)+=(*this)(i,k)*b(k,j);
        }
        return c;
    }
    Affine inverse() const {
        for(double v:values) if(!std::isfinite(v)) throw std::invalid_argument("nonfinite affine");
        if(values[12]!=0 || values[13]!=0 || values[14]!=0 || values[15]!=1)
            throw std::invalid_argument("expected affine homogeneous row");
        const double a=values[0],b=values[1],c=values[2],d=values[4],e=values[5],f=values[6],g=values[8],h=values[9],i=values[10];
        const double det=a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g);
        if(det==0 || !std::isfinite(det))throw std::invalid_argument("singular affine");
        Affine out;
        out(0,0)=(e*i-f*h)/det;out(0,1)=(c*h-b*i)/det;out(0,2)=(b*f-c*e)/det;
        out(1,0)=(f*g-d*i)/det;out(1,1)=(a*i-c*g)/det;out(1,2)=(c*d-a*f)/det;
        out(2,0)=(d*h-e*g)/det;out(2,1)=(b*g-a*h)/det;out(2,2)=(a*e-b*d)/det;
        const auto t=out.apply({values[3],values[7],values[11]},true);
        for(int k=0;k<3;++k)out(k,3)=-t[k];
        for(double v:out.values)if(!std::isfinite(v))throw std::invalid_argument("affine inverse is not representable");
        return out;
    }
    static Affine fromJson(const nlohmann::json& rows) {
        if(!rows.is_array() || (rows.size()!=3 && rows.size()!=4)) throw std::invalid_argument("expected 3x4 or 4x4 affine");
        Affine a;
        for(size_t r=0;r<rows.size();++r) {
            if(!rows[r].is_array() || rows[r].size()!=4)throw std::invalid_argument("expected affine row of four numbers");
            for(int c=0;c<4;++c) {
                if(!rows[r][c].is_number())throw std::invalid_argument("expected numeric affine coefficient");
                a(r,c)=rows[r][c].get<double>();
            }
        }
        (void)a.inverse();return a;
    }
    // A transform.json: {"transformation_matrix": 3x4 or 4x4 rows}.
    static Affine fromTransformJson(const nlohmann::json& json) {
        if(!json.is_object() || !json.contains("transformation_matrix"))
            throw std::invalid_argument("transform.json is missing transformation_matrix");
        return fromJson(json.at("transformation_matrix"));
    }
    static Affine loadTransformJson(const std::filesystem::path& path) {
        std::ifstream in(path);
        if(!in) throw std::invalid_argument("cannot open transform file: " + path.string());
        try { return fromTransformJson(nlohmann::json::parse(in)); }
        catch(const std::exception& e) { throw std::invalid_argument(path.string() + ": " + e.what()); }
    }
    // Of the linear part.
    double determinant() const {
        const auto& m=*this;
        return m(0,0)*(m(1,1)*m(2,2)-m(1,2)*m(2,1))-m(0,1)*(m(1,0)*m(2,2)-m(1,2)*m(2,0))+m(0,2)*(m(1,0)*m(2,1)-m(1,1)*m(2,0));
    }
    // The linear part is exactly a positive uniform scale: directions and
    // normals keep their exact values and lengths scale by (0,0).
    bool isUniformScale() const {
        for(int r=0;r<3;++r) for(int c=0;c<3;++c) if(r!=c && (*this)(r,c)!=0) return false;
        return (*this)(0,0)>0 && (*this)(0,0)==(*this)(1,1) && (*this)(1,1)==(*this)(2,2);
    }
    // The scale of a linear part that is a rotation (or reflection) times a
    // uniform scale: its columns orthogonal and of equal squared length within
    // `tolerance` relative to the first's. None otherwise.
    std::optional<double> similarityScale(double tolerance=1e-12) const {
        const auto column=[&](int c){return std::array<double,3>{(*this)(0,c),(*this)(1,c),(*this)(2,c)};};
        const auto dot=[](const auto& a,const auto& b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];};
        const auto x=column(0),y=column(1),z=column(2);
        const double squared=dot(x,x),bound=tolerance*squared;
        if(squared>0&&std::abs(dot(y,y)-squared)<=bound&&std::abs(dot(z,z)-squared)<=bound&&
           std::abs(dot(x,y))<=bound&&std::abs(dot(x,z))<=bound&&std::abs(dot(y,z))<=bound)
            return std::sqrt(squared);
        return std::nullopt;
    }
    // The factor lengths scale by: the uniform scale exactly, otherwise the
    // cube root of the determinant's magnitude.
    double lengthScale() const { return isUniformScale() ? (*this)(0,0) : std::cbrt(std::abs(determinant())); }
    // A direction through the linear part and a normal through its inverse
    // transpose, both returned unit length (zero stays zero). A uniform scale
    // returns them unchanged.
    std::array<double,3> mapDirection(const std::array<double,3>& d) const {
        if(isUniformScale()) return d;
        return unit(apply(d,true));
    }
    std::array<double,3> mapNormal(const std::array<double,3>& n) const {
        if(isUniformScale()) return n;
        return unit(normalTransform().apply(n,true));
    }
    // The linear part's inverse transpose, which maps normals (unnormalized);
    // compute it once to map many normals with mapDirection.
    Affine normalTransform() const {
        const auto inv=inverse();
        Affine t;
        for(int r=0;r<3;++r) for(int c=0;c<3;++c) t(r,c)=inv(c,r);
        return t;
    }
    nlohmann::json json() const {
        nlohmann::json rows=nlohmann::json::array();
        for(int r=0;r<3;++r) rows.push_back({(*this)(r,0),(*this)(r,1),(*this)(r,2),(*this)(r,3)});
        return rows;
    }
    bool equivalent(const Affine& b) const {
        for(int i=0;i<12;++i)if(std::abs(values[i]-b.values[i])>1e-10*std::max({1.,std::abs(values[i]),std::abs(b.values[i])}))return false;
        return true;
    }
    bool operator==(const Affine&) const = default;
private:
    static std::array<double,3> unit(const std::array<double,3>& v) {
        const double n=std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
        if(!(n>0)) return v;
        const double s=1./n;
        return {v[0]*s,v[1]*s,v[2]*s};
    }
};
}
