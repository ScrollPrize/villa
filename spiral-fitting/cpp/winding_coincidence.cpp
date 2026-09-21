#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>

namespace nb = nanobind;
namespace {
using V2 = std::array<double, 2>;
using V3 = std::array<double, 3>;
using Face = std::array<int, 3>;
double cross(V2 a, V2 b) { return a[0]*b[1]-a[1]*b[0]; }
V2 sub(V2 a, V2 b) { return {a[0]-b[0], a[1]-b[1]}; }
struct Box { V2 lo{INFINITY,INFINITY}, hi{-INFINITY,-INFINITY}; };
void extend(Box& b, V2 p) {
    for(int k=0;k<2;++k) { b.lo[k]=std::min(b.lo[k],p[k]); b.hi[k]=std::max(b.hi[k],p[k]); }
}
bool overlaps(const Box& a, const Box& b, double shift) {
    return a.lo[0]+shift<=b.hi[0]+1e-9 && b.lo[0]<=a.hi[0]+shift+1e-9 &&
           a.lo[1]<=b.hi[1]+1e-9 && b.lo[1]<=a.hi[1]+1e-9;
}
struct Node { Box box; int begin, end, left=-1, right=-1; };
int build_index(const std::vector<Box>& boxes, std::vector<int>& order,
                std::vector<Node>& nodes, int begin, int end) {
        Box b; for(int i=begin;i<end;++i) { extend(b,boxes[order[i]].lo); extend(b,boxes[order[i]].hi); }
        int n=int(nodes.size()); nodes.push_back({b,begin,end});
        if(end-begin>8) {
            int axis=(b.hi[1]-b.lo[1])>(b.hi[0]-b.lo[0]), mid=(begin+end)/2;
            std::nth_element(order.begin()+begin,order.begin()+mid,order.begin()+end,[&](int a,int c) {
                double x=boxes[a].lo[axis]+boxes[a].hi[axis],y=boxes[c].lo[axis]+boxes[c].hi[axis];
                return x==y?a<c:x<y;
            });
            int left=build_index(boxes,order,nodes,begin,mid),right=build_index(boxes,order,nodes,mid,end); nodes[n].left=left; nodes[n].right=right;
        }
        return n;
    }
struct Mesh {
    std::vector<V2> uv;
    std::vector<V3> xyz;
    std::vector<Face> faces;
    std::vector<Box> boxes;
    std::vector<int> order;
    std::vector<Node> nodes;
    using Doubles=nb::ndarray<const double,nb::ndim<2>,nb::c_contig,nb::device::cpu>;
    using Ints=nb::ndarray<const int,nb::ndim<2>,nb::c_contig,nb::device::cpu>;
    Mesh(Doubles coords, Doubles points, Ints triangles) {
        if(coords.shape(1)!=2 || points.shape(1)!=3 || triangles.shape(1)!=3 ||
           coords.shape(0)!=points.shape(0) || !triangles.shape(0))
            throw std::invalid_argument("Invalid coincidence mesh dimensions");
        for(size_t i=0;i<coords.shape(0);++i) {
            uv.push_back({coords(i,0),coords(i,1)}); xyz.push_back({points(i,0),points(i,1),points(i,2)});
            for(double x:uv.back()) if(!std::isfinite(x)) throw std::invalid_argument("Nonfinite parameter coordinate");
            for(double x:xyz.back()) if(!std::isfinite(x)) throw std::invalid_argument("Nonfinite source coordinate");
        }
        for(size_t i=0;i<triangles.shape(0);++i) {
            Face f{triangles(i,0),triangles(i,1),triangles(i,2)};
            for(int v:f) if(v<0 || size_t(v)>=uv.size()) throw std::invalid_argument("Invalid triangle index");
            if(std::abs(cross(sub(uv[f[1]],uv[f[0]]),sub(uv[f[2]],uv[f[0]])))<1e-12)
                throw std::invalid_argument("Degenerate theta/height triangle");
            faces.push_back(f); Box b; for(int v:f) extend(b,uv[v]); boxes.push_back(b);
        }
        order.resize(faces.size()); std::iota(order.begin(),order.end(),0); build_index(boxes,order,nodes,0,int(order.size()));
    }

};
V3 bary(V2 p,const std::array<V2,3>& t) {
    V2 a=sub(t[1],t[0]),b=sub(t[2],t[0]),q=sub(p,t[0]);
    double det=cross(a,b),u=cross(q,b)/det,v=cross(a,q)/det;
    return {1-u-v,u,v};
}
bool inside(V3 w) { return *std::min_element(w.begin(),w.end())>=-1e-10; }
template<class Check>
bool intersections(const std::array<V2,3>& ta, const std::array<V2,3>& tb, Check check) {
            for(auto p:ta) if(check(p)) return true;
            for(auto p:tb) if(check(p)) return true;
            for(int i=0;i<3;++i) for(int j=0;j<3;++j) {
                auto p=ta[i],q=tb[j],r=sub(ta[(i+1)%3],p),s=sub(tb[(j+1)%3],q);double den=cross(r,s);
                if(den==0) continue; // Collinear intersections have contained endpoints.
                double u=cross(sub(q,p),s)/den,v=cross(sub(q,p),r)/den;
                if(u>=0 && u<=1 && v>=0 && v<=1 && check({p[0]+u*r[0],p[1]+u*r[1]})) return true;
            }
    return false;
}
struct Result {
    bool conflict=false; double maximum=0; size_t pairs=0,vertices=0;
    int a=-1,b=-1; V2 point{}; V3 first{},second{};
};
// XYZ differences are affine on a triangle intersection. Their norm attains
// its maximum at an intersection vertex: contained corners or crossing edges.
Result compare(const Mesh& a,const Mesh& b,double shift,double tolerance,bool self) {
    Result out; std::vector<std::pair<int,int>> stack{{0,0}};
    while(!stack.empty()) {
        auto [na,nb]=stack.back(); stack.pop_back(); const auto& aa=a.nodes[na]; const auto& bb=b.nodes[nb];
        if(!overlaps(aa.box,bb.box,shift)) continue;
        if(aa.left>=0 || bb.left>=0) {
            if(aa.left>=0 && (bb.left<0 || aa.end-aa.begin>=bb.end-bb.begin)) {
                stack.emplace_back(aa.right,nb); stack.emplace_back(aa.left,nb);
            } else { stack.emplace_back(na,bb.right); stack.emplace_back(na,bb.left); }
            continue;
        }
        for(int ia=aa.begin;ia<aa.end;++ia) for(int ib=bb.begin;ib<bb.end;++ib) {
            int fa=a.order[ia],fb=b.order[ib];
            if(self && fa>=fb) continue;
            if(!overlaps(a.boxes[fa],b.boxes[fb],shift)) continue;
            ++out.pairs;
            std::array<V2,3> ta,tb;
            for(int k=0;k<3;++k) { ta[k]=a.uv[a.faces[fa][k]];ta[k][0]+=shift;tb[k]=b.uv[b.faces[fb][k]]; }
            auto check=[&](V2 p) {
                auto wa=bary(p,ta),wb=bary(p,tb);
                if(!inside(wa)||!inside(wb)) return false;
                V3 xa{},xb{};
                for(int j=0;j<3;++j) for(int k=0;k<3;++k) {
                    xa[k]+=wa[j]*a.xyz[a.faces[fa][j]][k];xb[k]+=wb[j]*b.xyz[b.faces[fb][j]][k];
                }
                double d=0;for(int k=0;k<3;++k) d+=(xa[k]-xb[k])*(xa[k]-xb[k]);d=std::sqrt(d);++out.vertices;
                if(d>out.maximum) {out.maximum=d;out.a=fa;out.b=fb;out.point=p;out.first=xa;out.second=xb;}
                out.conflict=d>tolerance;return out.conflict;
            };
            if(intersections(ta,tb,check)) return out;
        }
    }
    return out;
}
nb::dict report(const Mesh& a,const Mesh& b,double shift,double tolerance,bool self) {
    if(!std::isfinite(shift)||!std::isfinite(tolerance)||tolerance<=0) throw std::invalid_argument("Invalid coincidence tolerance/shift");
    Result r; { nb::gil_scoped_release release; r=compare(a,b,shift,tolerance,self); }
    nb::dict d;d["conflict"]=r.conflict;d["max_distance"]=r.maximum;d["triangle_pairs"]=r.pairs;d["intersection_vertices"]=r.vertices;
    d["face_a"]=r.a;d["face_b"]=r.b;d["parameter"]=r.point;d["xyz_a"]=r.first;d["xyz_b"]=r.second;return d;
}
V3 sub3(V3 a,V3 b) { return {a[0]-b[0],a[1]-b[1],a[2]-b[2]}; }
double dot(V3 a,V3 b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
V3 cross3(V3 a,V3 b) { return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]}; }
V3 unit(V3 a) {
    double length=std::sqrt(dot(a,a));
    if(!std::isfinite(length)||length<1e-12) throw std::invalid_argument("Undefined local surface normal");
    for(double& x:a) x/=length;
    return a;
}
// Each window consists of the eight triangles in a 3x3 vertex neighborhood,
// or two triangles in a 2x2 fallback. Local normals are fitted separately so
// between-surface displacement cannot tilt the measurement plane.
struct ThicknessMesh {
    Mesh mesh;
    std::vector<std::vector<int>> windows;
    std::vector<V3> normals;
    std::vector<Box> boxes;
    std::vector<int> order;
    std::vector<Node> nodes;
    ThicknessMesh(Mesh::Doubles uv,Mesh::Doubles xyz,Mesh::Ints faces,
                  Mesh::Ints window_faces,Mesh::Doubles normal):mesh(uv,xyz,faces) {
        if(!window_faces.shape(0)||window_faces.shape(1)!=8 || normal.shape(1)!=3 ||
           normal.shape(0)!=window_faces.shape(0)) throw std::invalid_argument("Invalid normal-grid windows");
        std::vector<bool> covered(mesh.faces.size(),false);
        for(size_t i=0;i<window_faces.shape(0);++i) {
            std::vector<int> ids; Box box;
            for(size_t j=0;j<8;++j) {
                int f=window_faces(i,j); if(f==-1) continue;
                if(f<0||size_t(f)>=mesh.faces.size()) throw std::invalid_argument("Invalid window face");
                if(std::find(ids.begin(),ids.end(),f)!=ids.end()) throw std::invalid_argument("Duplicate window face");
                ids.push_back(f); covered[f]=true; extend(box,mesh.boxes[f].lo);extend(box,mesh.boxes[f].hi);
            }
            if(ids.size()!=2 && ids.size()!=8) throw std::invalid_argument("Window must be 2x2 or 3x3");
            windows.push_back(ids);boxes.push_back(box);
            normals.push_back(unit({normal(i,0),normal(i,1),normal(i,2)}));
        }
        if(std::find(covered.begin(),covered.end(),false)!=covered.end())
            throw std::invalid_argument("Normal-grid windows leave uncovered faces");
        order.resize(windows.size());std::iota(order.begin(),order.end(),0);
        build_index(boxes,order,nodes,0,int(order.size()));
    }
};
struct ThicknessResult:Result {
    size_t windows=0; V3 normal{};
    bool incompatible_normal=false;
};
ThicknessResult thickness(const ThicknessMesh& a,const ThicknessMesh& b,
                          double shift,double tolerance,double min_cos) {
    ThicknessResult out;std::vector<std::pair<int,int>> stack{{0,0}};
    while(!stack.empty()) {
        auto [na,nb]=stack.back();stack.pop_back();const auto& aa=a.nodes[na];const auto& bb=b.nodes[nb];
        if(!overlaps(aa.box,bb.box,shift)) continue;
        if(aa.left>=0 || bb.left>=0) {
            if(aa.left>=0 && (bb.left<0 || aa.end-aa.begin>=bb.end-bb.begin)) {
                stack.emplace_back(aa.right,nb);stack.emplace_back(aa.left,nb);
            } else {stack.emplace_back(na,bb.right);stack.emplace_back(na,bb.left);}
            continue;
        }
        for(int ia=aa.begin;ia<aa.end;++ia) for(int ib=bb.begin;ib<bb.end;++ib) {
            int wa=a.order[ia],wb=b.order[ib];
            if(!overlaps(a.boxes[wa],b.boxes[wb],shift)) continue;
            V3 n=a.normals[wa],m=b.normals[wb];double cosine=dot(n,m);
            if(cosine<0) for(double& x:m) x=-x;
            for(int k=0;k<3;++k) n[k]+=m[k];n=unit(n);
            V3 axis{1,0,0};if(std::abs(n[0])>.8) axis={0,1,0};
            V3 u=unit(cross3(n,axis)),v=cross3(n,u);
            V3 origin=a.mesh.xyz[a.mesh.faces[a.windows[wa][0]][0]];
            auto project=[&](V3 p) {p=sub3(p,origin);return V2{dot(p,u),dot(p,v)};};
            double low=INFINITY,high=-INFINITY;V3 low_xyz{},high_xyz{};V2 witness{};
            size_t before=out.vertices;
            for(int fa:a.windows[wa]) for(int fb:b.windows[wb]) {
                std::array<V2,3> ta,tb;
                for(int j=0;j<3;++j) {
                    ta[j]=project(a.mesh.xyz[a.mesh.faces[fa][j]]);
                    tb[j]=project(b.mesh.xyz[b.mesh.faces[fb][j]]);
                }
                if(std::abs(cross(sub(ta[1],ta[0]),sub(ta[2],ta[0])))<1e-12 ||
                   std::abs(cross(sub(tb[1],tb[0]),sub(tb[2],tb[0])))<1e-12)
                    throw std::invalid_argument("Degenerate triangle in local tangent plane");
                ++out.pairs;
                intersections(ta,tb,[&](V2 p) {
                    V3 ba=bary(p,ta),bb=bary(p,tb);if(!inside(ba)||!inside(bb)) return false;
                    V3 xa{},xb{};V2 pa{},pb{};
                    for(int j=0;j<3;++j) {
                        int va=a.mesh.faces[fa][j],vb=b.mesh.faces[fb][j];
                        for(int k=0;k<3;++k) {xa[k]+=ba[j]*a.mesh.xyz[va][k];xb[k]+=bb[j]*b.mesh.xyz[vb][k];}
                        for(int k=0;k<2;++k) {pa[k]+=ba[j]*a.mesh.uv[va][k];pb[k]+=bb[j]*b.mesh.uv[vb][k];}
                    }
                    // Alignment may change local theta slightly, never by a
                    // full winding. The window BVH alone is not enough at a seam.
                    if(std::abs(pa[0]+shift-pb[0])>=5000.) return false;
                    ++out.vertices;witness=pb;
                    for(V3 x:{xa,xb}) {
                        double h=dot(sub3(x,origin),n);
                        if(h<low) {low=h;low_xyz=x;}
                        if(h>high) {high=h;high_xyz=x;}
                    }
                    return false;
                });
            }
            if(out.vertices==before) continue;
            ++out.windows;double spread=high-low;
            if(spread>out.maximum || std::abs(cosine)<min_cos) {
                out.maximum=std::max(out.maximum,spread);out.a=wa;out.b=wb;out.point=witness;
                out.first=low_xyz;out.second=high_xyz;out.normal=n;
            }
            out.incompatible_normal=std::abs(cosine)<min_cos;
            out.conflict=spread>tolerance || out.incompatible_normal;
            if(out.conflict) return out;
        }
    }
    return out;
}
nb::dict thickness_report(const ThicknessMesh& a,const ThicknessMesh& b,double shift,
                          double tolerance,double min_cos) {
    if(!std::isfinite(shift)||!std::isfinite(tolerance)||tolerance<=0 ||
       !std::isfinite(min_cos)||min_cos<0 ||min_cos>1) throw std::invalid_argument("Invalid thickness settings");
    ThicknessResult r;
    {nb::gil_scoped_release release;r=thickness(a,b,shift,tolerance,min_cos);}
    nb::dict d;d["conflict"]=r.conflict;d["max_distance"]=r.maximum;
    d["triangle_pairs"]=r.pairs;d["intersection_vertices"]=r.vertices;d["window_pairs"]=r.windows;
    d["window_a"]=r.a;d["window_b"]=r.b;d["parameter"]=r.point;
    d["xyz_min"]=r.first;d["xyz_max"]=r.second;d["normal"]=r.normal;
    d["incompatible_normal"]=r.incompatible_normal;return d;
}

}
NB_MODULE(winding_coincidence,m) {
    nb::class_<ThicknessMesh>(m,"ThicknessMesh")
        .def(nb::init<Mesh::Doubles,Mesh::Doubles,Mesh::Ints,Mesh::Ints,Mesh::Doubles>())
        .def("compare",[](const ThicknessMesh& a,const ThicknessMesh& b,double shift,double tolerance,double min_cos){return thickness_report(a,b,shift,tolerance,min_cos);},nb::arg("other"),nb::arg("shift"),nb::arg("tolerance"),nb::arg("min_cos")=0.);

    nb::class_<Mesh>(m,"Mesh")
        .def(nb::init<Mesh::Doubles,Mesh::Doubles,Mesh::Ints>())
        .def("compare",[](const Mesh& a,const Mesh& b,double shift,double tolerance){return report(a,b,shift,tolerance,false);})
        .def("check_self",[](const Mesh& a,double tolerance){return report(a,a,0,tolerance,true);});
}
