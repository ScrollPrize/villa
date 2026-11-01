#pragma once
#include "vc/fiber_tracer/detail/PortableTraceMath.hpp"

namespace vc::fiber_tracer::differential {
// Small forward differential for ONE local polyline term, not a trace replay.
// Coordinates of three adjacent vertices and incoming direction history give
// twelve variables. Cross-term/history adjoints are gathered separately.
template<int N> struct Scalar {
    float value=0, gradient[N]{};
    VC_TRACE_HD Scalar()=default;
    VC_TRACE_HD Scalar(float v):value(v){}
    VC_TRACE_HD static Scalar variable(float v,int axis) {Scalar s(v);s.gradient[axis]=1;return s;}
    VC_TRACE_HD Scalar operator+(Scalar b)const {Scalar r(value+b.value);for(int i=0;i<N;++i)r.gradient[i]=gradient[i]+b.gradient[i];return r;}
    VC_TRACE_HD Scalar operator-(Scalar b)const {Scalar r(value-b.value);for(int i=0;i<N;++i)r.gradient[i]=gradient[i]-b.gradient[i];return r;}
    VC_TRACE_HD Scalar operator-()const {return Scalar(0)-*this;}
    VC_TRACE_HD Scalar operator*(Scalar b)const {Scalar r(value*b.value);for(int i=0;i<N;++i)r.gradient[i]=gradient[i]*b.value+value*b.gradient[i];return r;}
    VC_TRACE_HD Scalar operator/(Scalar b)const {Scalar r(value/b.value);for(int i=0;i<N;++i)r.gradient[i]=(gradient[i]-r.value*b.gradient[i])/b.value;return r;}
    VC_TRACE_HD Scalar& operator+=(Scalar b){return *this=*this+b;}
    VC_TRACE_HD Scalar& operator*=(Scalar b){return *this=*this*b;}
    VC_TRACE_HD bool operator<(Scalar b)const{return value<b.value;}
    VC_TRACE_HD bool operator>(Scalar b)const{return value>b.value;}
    VC_TRACE_HD bool operator<=(Scalar b)const{return value<=b.value;}
    VC_TRACE_HD bool operator>=(Scalar b)const{return value>=b.value;}
    VC_TRACE_HD friend Scalar operator+(float a,Scalar b){return Scalar(a)+b;}
    VC_TRACE_HD friend Scalar operator-(float a,Scalar b){return Scalar(a)-b;}
    VC_TRACE_HD friend Scalar operator*(float a,Scalar b){return Scalar(a)*b;}
    VC_TRACE_HD friend Scalar operator/(float a,Scalar b){return Scalar(a)/b;}
};
template<int N> VC_TRACE_HD Scalar<N> unary(Scalar<N> a,float v,float derivative){Scalar<N> r(v);for(int i=0;i<N;++i)r.gradient[i]=a.gradient[i]*derivative;return r;}
template<int N> VC_TRACE_HD Scalar<N> sqrt(Scalar<N> a){float v=std::sqrt(a.value);return unary(a,v,v>0?.5f/v:0);}
template<int N> VC_TRACE_HD Scalar<N> abs(Scalar<N> a){return unary(a,std::abs(a.value),a.value>0?1:a.value<0?-1:0);}
template<int N> VC_TRACE_HD Scalar<N> acos(Scalar<N> a){return unary(a,std::acos(a.value),std::abs(a.value)<1?-1/std::sqrt(1-a.value*a.value):0);}
template<int N> VC_TRACE_HD Scalar<N> asin(Scalar<N> a){return unary(a,std::asin(a.value),std::abs(a.value)<1?1/std::sqrt(1-a.value*a.value):0);}
template<int N> VC_TRACE_HD bool isfinite(Scalar<N> a){return std::isfinite(a.value);}
template<class S> struct Vector {
    S x{},y{},z{};
    VC_TRACE_HD Vector()=default;
    VC_TRACE_HD Vector(S a,S b,S c):x(a),y(b),z(c){}
    VC_TRACE_HD S& operator[](int i){return (&x)[i];}
    VC_TRACE_HD S operator[](int i)const{return (&x)[i];}
    VC_TRACE_HD Vector operator+(Vector b)const{return {x+b.x,y+b.y,z+b.z};}
    VC_TRACE_HD Vector operator-(Vector b)const{return {x-b.x,y-b.y,z-b.z};}
    VC_TRACE_HD Vector operator*(S s)const{return {x*s,y*s,z*s};}
    VC_TRACE_HD Vector operator/(S s)const{return {x/s,y/s,z/s};}
    VC_TRACE_HD S dot(Vector b)const {S v=0;v+=x*b.x;v+=y*b.y;v+=z*b.z;return v;}
};
}
