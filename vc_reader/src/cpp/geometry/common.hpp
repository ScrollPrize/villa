#pragma once
#include "math.hpp"
#include <cstddef>
#include <limits>
namespace vcr::geometry {
struct Node { V lo,hi; int left=-1,right=-1,begin=0,end=0; };
struct CurvesView {
    const float* points=nullptr; const float* arcs=nullptr; const std::int64_t* offsets=nullptr;
    const int* owners=nullptr; const Node* nodes=nullptr; const int* order=nullptr; const int* roots=nullptr;
    int curves=0,root=-1; bool vertices=false;
};
VCR_GEOMETRY_HD inline bool curve_ok(const CurvesView& c,std::int64_t id){return id>=0&&id<c.curves;}
VCR_GEOMETRY_HD inline void gather(const CurvesView& c,std::int64_t index,bool segments,float* out,float* arcs,std::int64_t* owner,bool* valid){
    int width=segments?2:1;for(int j=0;j<width;++j){store(out+3*j,{NAN,NAN,NAN});arcs[j]=NAN;}
    *owner=-1;*valid=false;
    if(index<0||index>=c.offsets[c.curves])return;
    int id=c.owners[index];auto next=index+1;
    if(segments&&next==c.offsets[id+1]){if(index!=c.offsets[id])return;next=index;}
    store(out,vector(c.points+std::size_t(3)*index));arcs[0]=c.arcs[index];*owner=id;*valid=true;
    if(segments){store(out+3,vector(c.points+std::size_t(3)*next));arcs[1]=c.arcs[next];}
}
VCR_GEOMETRY_HD inline int segment(const CurvesView& c,int first,int last,float s){
    int lo=first,hi=last;
    while(lo<hi){int mid=lo+(hi-lo)/2;if(c.arcs[mid]<=s)lo=mid+1;else hi=mid;}
    return lo>first ? (lo-1<last?lo-1:last-1) : first;
}
VCR_GEOMETRY_HD inline V at(const CurvesView& c,int a,int end,float s){
    if(a>=end)return vector(c.points+std::size_t(3)*end);
    float span=c.arcs[a+1]-c.arcs[a];float t=span>0?clamp((s-c.arcs[a])/span):0;
    return vector(c.points+std::size_t(3)*a)*(1-t)+vector(c.points+std::size_t(3)*(a+1))*t;
}
VCR_GEOMETRY_HD inline void interpolate(const CurvesView& c,std::int64_t id,float s,float* out,bool* valid){
    store(out,{NAN,NAN,NAN});*valid=false;
    if(!curve_ok(c,id)||!std::isfinite(s))return;
    int first=int(c.offsets[id]),last=int(c.offsets[id+1])-1;
    *valid=s>=c.arcs[first]&&s<=c.arcs[last];
    s=fminf(c.arcs[last],fmaxf(c.arcs[first],s));
    if(first==last){store(out,vector(c.points+std::size_t(3)*first));return;}
    if(s==c.arcs[last]){store(out,vector(c.points+std::size_t(3)*last));return;}
    store(out,at(c,segment(c,first,last,s),last,s));
}
VCR_GEOMETRY_HD inline void intersect(const CurvesView& c,std::int64_t id,const float* plane,const float* interval,
    int direction,float* out,float* arc,std::int64_t* index,bool* valid){
    store(out,{NAN,NAN,NAN});*arc=NAN;*index=-1;*valid=false;
    V normal=vector(plane);
    if(!curve_ok(c,id)||!finite(normal)||!std::isfinite(plane[3]))return;
    float scale=fmaxf(fabsf(normal.x),fmaxf(fabsf(normal.y),fabsf(normal.z))),offset=plane[3];
    if(!(scale>0))return;
    if(scale>1e10f||scale<1e-10f){normal={normal.x/scale,normal.y/scale,normal.z/scale};offset/=scale;}
    if(!std::isfinite(offset))return;
    int first=int(c.offsets[id]),last=int(c.offsets[id+1])-1;
    if(first==last)return;
    float start=interval?interval[0]:c.arcs[first],stop=interval?interval[1]:c.arcs[last];
    if(!std::isfinite(start)||!std::isfinite(stop))return;
    start=fminf(c.arcs[last],fmaxf(c.arcs[first],start));stop=fminf(c.arcs[last],fmaxf(c.arcs[first],stop));
    if(start==stop)return;
    bool forward=stop>start;
    int seg=segment(c,first,last,start);
    // A reverse traversal beginning exactly on a vertex starts on its preceding segment.
    if(!forward&&seg>first&&c.arcs[seg]==start)--seg;
    for(;seg>=first&&seg<last;seg+=forward?1:-1){
        float a=forward?fmaxf(start,c.arcs[seg]):fminf(start,c.arcs[seg+1]);
        float b=forward?fminf(stop,c.arcs[seg+1]):fmaxf(stop,c.arcs[seg]);
        if(forward?a>b:a<b)break;
        if(a==b)continue;
        V p=at(c,seg,last,a),q=at(c,seg,last,b);
        float u=dot(normal,p)-offset,v=dot(normal,q)-offset;
        bool hit=(direction>=0&&u<0&&v>=0)||(direction<=0&&u>0&&v<=0);
        if(hit){float t=u/(u-v);store(out,p*(1-t)+q*t);*arc=a+(b-a)*t;*index=seg;*valid=true;return;}
        if(b==stop)break;
    }
}
VCR_GEOMETRY_HD inline float box_distance(V p,const Node& n){
    float x=fmaxf(0,fmaxf(n.lo.x-p.x,p.x-n.hi.x));
    float y=fmaxf(0,fmaxf(n.lo.y-p.y,p.y-n.hi.y));
    float z=fmaxf(0,fmaxf(n.lo.z-p.z,p.z-n.hi.z));return x*x+y*y+z*z;
}
VCR_GEOMETRY_HD inline void closest(const CurvesView& c,V p,std::int64_t curve,float limit,float* out,float* distance,
    float* arc,std::int64_t* index,std::int64_t* owner,bool* valid){
    store(out,{NAN,NAN,NAN});*distance=INFINITY;*arc=NAN;*index=*owner=-1;*valid=false;
    if(!finite(p)||curve< -1||curve>=c.curves||!(limit>=0))return;
    int root=curve<0?c.root:c.roots[curve];if(root<0)return;
    float best=limit*limit;int stack[64],top=0;stack[top++]=root;
    while(top){
        const Node& n=c.nodes[stack[--top]];
        if(box_distance(p,n)>best)continue;
        if(n.left>=0){
            float a=box_distance(p,c.nodes[n.left]),b=box_distance(p,c.nodes[n.right]);
            int near=n.left,far=n.right;if(a>b){near=n.right;far=n.left;float t=a;a=b;b=t;}
            if(b<=best)stack[top++]=far;if(a<=best)stack[top++]=near;
        }else for(int j=n.begin;j<n.end;++j){
            int i=c.order[j],id=c.owners[i];V a=vector(c.points+std::size_t(3)*i),q=a;float t=0;
            if(!c.vertices&&i+1<c.offsets[id+1])q=closest_segment(p,a,vector(c.points+std::size_t(3)*(i+1)),t);
            V d=p-q;float dd=dot(d,d);
            if(std::isfinite(dd)&&(dd<best||(dd==best&&(*index<0||i<*index)))){
                best=dd;store(out,q);*distance=sqrtf(dd);*index=i;*owner=id;
                *arc=c.arcs[i]+(t>0?t*(c.arcs[i+1]-c.arcs[i]):0);*valid=true;
            }
        }
    }
}
VCR_GEOMETRY_HD inline bool overlaps(const Node& n,V lo,V hi){
    return n.hi.x>=lo.x&&n.lo.x<=hi.x&&n.hi.y>=lo.y&&n.lo.y<=hi.y&&n.hi.z>=lo.z&&n.lo.z<=hi.z;
}
VCR_GEOMETRY_HD inline void query_box(const CurvesView& c,V lo,V hi,std::int64_t curve,int capacity,std::int64_t* indices,std::int64_t* count,bool* valid){
    for(int j=0;j<capacity;++j)indices[j]=-1;
    *count=0;*valid=false;
    if(!finite(lo)||!finite(hi)||lo.x>hi.x||lo.y>hi.y||lo.z>hi.z||curve< -1||curve>=c.curves)return;
    *valid=true;int root=curve<0?c.root:c.roots[curve];if(root<0)return;
    int stack[64],top=0;stack[top++]=root;
    while(top){const Node& n=c.nodes[stack[--top]];if(!overlaps(n,lo,hi))continue;
        if(n.left>=0){stack[top++]=n.right;stack[top++]=n.left;}
        else for(int j=n.begin;j<n.end;++j){
            int i=c.order[j];V a=vector(c.points+std::size_t(3)*i),b=a;
            if(!c.vertices&&i+1<c.offsets[c.owners[i]+1])b=vector(c.points+std::size_t(3)*(i+1));
            Node bounds{{fminf(a.x,b.x),fminf(a.y,b.y),fminf(a.z,b.z)},
                        {fmaxf(a.x,b.x),fmaxf(a.y,b.y),fmaxf(a.z,b.z)}};
            if(overlaps(bounds,lo,hi)){if(*count<capacity)indices[*count]=i;++*count;}
        }
    }
}
struct VoxelGrid { std::int64_t shape[3],batches,volume; };
constexpr std::int64_t missing_key=INT64_MAX;
VCR_GEOMETRY_HD inline std::int64_t voxel_key(V p,std::int64_t batch,const VoxelGrid& g){
    if(batch<0||batch>=g.batches||!finite(p))return missing_key;
    float f[3]{p.x,p.y,p.z};std::int64_t cell[3];
    for(int a=0;a<3;++a){if(f[a]<-.5f||f[a]>float(g.shape[a])-.5f)return missing_key;
        cell[a]=nearest(f[a],0);if(cell[a]<0||cell[a]>=g.shape[a])return missing_key;}
    return batch*g.volume+(cell[0]*g.shape[1]+cell[1])*g.shape[2]+cell[2];
}
VCR_GEOMETRY_HD inline bool contains(const std::int64_t* keys,std::size_t count,std::int64_t key){
    if(key==missing_key)return false;std::size_t lo=0,hi=count;
    while(lo<hi){auto mid=lo+(hi-lo)/2;if(keys[mid]<key)lo=mid+1;else hi=mid;}
    return lo<count&&keys[lo]==key;
}
}
