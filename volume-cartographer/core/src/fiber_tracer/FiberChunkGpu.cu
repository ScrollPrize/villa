// Optional resident CUDA chunk backend and standalone fixture benchmark.
// One block owns each seed half; fields, search state and paths stay on device.
#include <cuda_runtime.h>
#include "vc/fiber_tracer/detail/PortableTraceMath.hpp"
#include "vc/core/util/PortableTensor.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <vector>
#include <stdexcept>

using Json=nlohmann::json;
using namespace vc::fiber_tracer;
struct V {
    float x=0,y=0,z=0;
    VC_TRACE_HD V()=default;
    VC_TRACE_HD V(float a,float b,float c):x(a),y(b),z(c){}
    VC_TRACE_HD float& operator[](int i){return (&x)[i];}
    VC_TRACE_HD float operator[](int i)const{return (&x)[i];}
    VC_TRACE_HD V operator+(V b)const{return {x+b.x,y+b.y,z+b.z};}
    VC_TRACE_HD V operator-(V b)const{return {x-b.x,y-b.y,z-b.z};}
    VC_TRACE_HD V operator*(float s)const{return {x*s,y*s,z*s};}
    VC_TRACE_HD V operator/(float s)const{return {x/s,y/s,z/s};}
    VC_TRACE_HD V& operator*=(float s){x*=s;y*=s;z*=s;return *this;}
    VC_TRACE_HD float dot(V b)const{float v=0;v+=x*b.x;v+=y*b.y;v+=z*b.z;return v;}
    VC_TRACE_HD V cross(V b)const{return {y*b.z-z*b.y,z*b.x-x*b.z,x*b.y-y*b.x};}
};
VC_TRACE_HD V unit(V a,V fallback={0,0,0}) {
    float n=std::sqrt(a.dot(a));return n>1e-6f&&std::isfinite(n)?a/n:fallback;
}
constexpr int MAX_BEAM=8,MAX_PARENT=32,MAX_DIR=81,MAX_CAND=MAX_PARENT*MAX_DIR;
constexpr int MAX_STEPS=160,MAX_NODES=(MAX_STEPS/2+1)*(MAX_PARENT+MAX_BEAM)+1;
struct Params {
    int shape[3],origin[3],dirs=0,steps=0,historySteps=0;
    double spacing=0,tolerance=0,cutoff=0;
    V lo,hi;
    float step=0,prune=0,cumulative=0;
    bool maskNormals=true,cosWidth=true;
    int refineIterations=0,refineWindow=0;
    float refineEpsilon=.25f,refineStep=1,refineCap=8;
    FiberLocalSmoothnessConfig smooth;
};
struct Sample {V dir,normal;float presence=0,width=0;bool valid=false;};
struct Fields {const unsigned char* bytes;const float* widths;const float* tensors;int* error;};
template<bool ScalarsOnly=false>
VC_TRACE_HD Sample sample(Fields f,Params p,V q,V hint) {
    int b[3];float frac[3];double df[3];
    for(int a=0;a<3;++a) {
        double scaled=double(q[a])/p.spacing;int origin=int(std::floor(scaled));
        b[a]=origin-p.origin[a];df[a]=scaled-origin;frac[a]=float(df[a]);
        if(b[a]<0||b[a]+1>=p.shape[a]) {
#ifdef __CUDA_ARCH__
            atomicExch(f.error,1);
#endif
            return {};
        }
    }
    float tensor[2][6]={},presence=0,validNormal=0;double ws[8],weighted=0,denom=0;
    bool all=true;
    for(int k=0;k<8;++k) {
        size_t i=(size_t(b[2]+((k>>2)&1))*p.shape[1]+b[1]+((k>>1)&1))*p.shape[0]+b[0]+(k&1);
        auto bytes=f.bytes+6*i;
        float w=((k&4)?frac[2]:1-frac[2])*((k&2)?frac[1]:1-frac[1])*((k&1)?frac[0]:1-frac[0]);
        presence=std::fma(w,float(bytes[0]),presence);
        validNormal=std::fma(w,float(bytes[3]),validNormal);
        ws[k]=f.widths[i];all&=bytes[0]>0;
        double dw=((k&1)?df[0]:1-df[0])*((k&2)?df[1]:1-df[1])*((k&4)?df[2]:1-df[2]);
        if(bytes[0]&&dw>0){weighted+=dw*ws[k];denom+=dw;}
        if constexpr(!ScalarsOnly) for(int axis=0;axis<2;++axis) {
            float weight=(axis==0?!bytes[0]:(p.maskNormals&&!bytes[3]))?0:w;
            const float* t=f.tensors+6*((unsigned(bytes[axis*3+1])<<8)|bytes[axis*3+2]);
            for(int j=0;j<6;++j)tensor[axis][j]=std::fma(weight,t[j],tensor[axis][j]);
        }
    }
    Sample s;s.presence=std::clamp(presence/255.f,0.f,1.f);
    if constexpr(!ScalarsOnly) {
        // Match native interpolation: the eigensolver's degenerate fallback
        // is not evidence of a direction when masked support is empty.
        if(tensor[0][0]+tensor[0][3]+tensor[0][5]>0)
            s.dir=vc::core::portable::principalFloatTensorAxis(tensor[0][0],tensor[0][1],tensor[0][2],tensor[0][3],tensor[0][4],tensor[0][5],hint);
        if(validNormal>0 && tensor[1][0]+tensor[1][3]+tensor[1][5]>0)
            s.normal=vc::core::portable::principalFloatTensorAxis(tensor[1][0],tensor[1][1],tensor[1][2],tensor[1][3],tensor[1][4],tensor[1][5],V{});
    }
    if(all) {
        double a=ws[0]*(1-df[0])+ws[1]*df[0],b0=ws[2]*(1-df[0])+ws[3]*df[0];
        double c=ws[4]*(1-df[0])+ws[5]*df[0],d=ws[6]*(1-df[0])+ws[7]*df[0];
        s.width=float((a*(1-df[1])+b0*df[1])*(1-df[2])+(c*(1-df[1])+d*df[1])*df[2]);
    } else s.width=denom>0?float(weighted/denom):0;
    s.valid=s.dir.dot(s.dir)>1e-12f;return s;
}
struct Node {V point;int previous=-1,depth=0;float modulated=0;double scoringWidth=0;
    V planeOrigin,planeNormal;bool planeFrozen=false;};
struct State {V point,previous,current,history;float loss=0,modulated=0;int node=0,historySamples=0,original=0;bool valid=true,presenceRejected=false;};
struct Seed {V point,direction;};
struct Output {V points[MAX_STEPS+1];float scores[MAX_STEPS+1];double widths[MAX_STEPS+1];
    V rejected;float rejectedScore=0;double rejectedWidth=0;bool hasRejected=false;
    int count=0,reason=0;unsigned long long candidates=0,scored=0,boundViolations=0;
    int refinementCommits=0,refinementAccepted=0,refinedFrontiers=0;float refinementGain=0,maxPlaneError=0,maxMovement=0;};
// Reason: 0 distance, 1 cutoff, 2 boundary, 3 no candidates, 4 invalid width,
// 5 device capacity/field error. Never silently truncate at an implementation cap.
VC_TRACE_HD bool outside(V q,Params p){for(int a=0;a<3;++a)if(q[a]<p.lo[a]||q[a]>=p.hi[a])return true;return false;}
VC_TRACE_HD bool better(const State& a,const State& b){return a.loss<b.loss||(a.loss==b.loss&&a.original<b.original);}
VC_TRACE_HD State scoreAtPoint(Fields fields,Params p,const State& parent,V point,V direction,double widthMean,int original,bool cutoff=true) {
    auto s=sample(fields,p,point,direction);
    State out;out.valid=s.valid;out.loss=INFINITY;out.original=original;
    if(!s.valid)return out;
    float scale=portable::widthPresenceScale(s.width,widthMean,p.tolerance,p.cosWidth);
    float local=portable::fiberLocalAlignmentLossInline(s.presence*scale,parent.previous,direction,parent.current,s.dir);
    auto smooth=portable::fiberLocalSmoothnessCostInline(parent.previous,direction,s.normal,s.normal.dot(s.normal)>1e-12f,p.smooth);
    float cumulative=0;
    if(p.cumulative>0 && s.normal.dot(s.normal)>1e-12f) {
        V h=unit(parent.history-s.normal*portable::clampFiberLocalUnit(parent.history.dot(s.normal)));
        V d=unit(direction-s.normal*portable::clampFiberLocalUnit(direction.dot(s.normal)));
        if(h.dot(h)>1e-12f&&d.dot(d)>1e-12f)cumulative=p.cumulative*portable::fiberLocalExcessAngleSquared(portable::fiberLocalAngleBetweenUnit(h,d),p.smooth.freeAngleRadians);
    }
    out.point=point;out.previous=direction;out.current=s.dir;out.modulated=1-local;
    out.loss=parent.loss+(local+smooth.total()+cumulative);
    if (cutoff && p.cutoff>0 && !(out.modulated >= p.cutoff)) {out.valid=false;out.presenceRejected=true;}
    out.history=parent.historySamples<=0||p.historySteps<=1?unit(direction):unit(unit(direction)+unit(parent.history)*float(std::clamp(parent.historySamples,1,p.historySteps-1)),unit(direction));
    out.historySamples=std::min(parent.historySamples+1,std::max(1,p.historySteps));
    out.node=parent.node;return out;
}
VC_TRACE_HD State candidate(Fields fields,Params p,const State& parent,float u,float v,double widthMean,int original) {
    V forward=unit(parent.current,{1,0,0});
    auto basis=portable::traceBasis(forward);V b0=basis[0],b1=basis[1];
    V direction=unit(forward+b0*u+b1*v,forward),point=parent.point+direction*p.step;
    return scoreAtPoint(fields,p,parent,point,direction,widthMean,original);
}

// All probes check the interpolation footprint before sampling: a rejected
// optimization trial is not an inference-cache miss.
VC_TRACE_HD bool inField(Params p,V q) {
    for(int a=0;a<3;++a){int b=int(std::floor(double(q[a])/p.spacing))-p.origin[a];if(b<0||b+1>=p.shape[a])return false;}
    return true;
}
VC_TRACE_HD State initialState(Seed seed) {
    State s;s.point=seed.point;s.previous=s.current=s.history=unit(seed.direction,{1,0,0});return s;
}
// Full history is replayed even when only the final N points can move. The
// optional single-point perturbation avoids private copies of the prefix.
VC_TRACE_HD State replayPoints(Fields f,Params p,Seed seed,const V* points,int n,double width,
                              int perturb=-1,V delta={},bool cutoff=true) {
    State s=initialState(seed);
    for(int i=1;i<n;++i){V q=points[i]+(i==perturb?delta:V{});
        if(!inField(p,q)||(q-s.point).dot(q-s.point)<1e-12f){s.valid=false;s.loss=INFINITY;return s;}
        s=scoreAtPoint(f,p,s,q,unit(q-s.point),width,0,cutoff);
        if(!s.valid){s.loss=INFINITY;return s;}
    }return s;
}
struct RefineScratch {
    V points[MAX_STEPS+1],trial[MAX_STEPS+1],gradient[MAX_STEPS+1];
    int ids[MAX_STEPS+1],n,first,accepted;
};
VC_TRACE_HD void freezePlanes(Node* nodes,const int* pending,int n) {
    // Freeze the whole common-prefix batch before any predecessor is moved.
    for(int i=0;i<n;++i){auto& node=nodes[pending[i]];if(!node.planeFrozen){
        node.planeOrigin=node.point;node.planeNormal=unit(node.point-nodes[node.previous].point);
        node.planeFrozen=true;
    }}
}
VC_TRACE_HD void refinementGradient(Fields f,Params p,Seed seed,Node* nodes,double widthMean,RefineScratch& work,int i) {
    work.gradient[i]={};auto& node=nodes[work.ids[i]];
    if(outside(node.point,p))return;
    auto axes=portable::traceBasis(node.planeNormal);
    for(int axis=0;axis<2;++axis){V d=axes[axis]*p.refineEpsilon;
        float plus=replayPoints(f,p,seed,work.points,work.n,widthMean,i,d,false).loss;
        float minus=replayPoints(f,p,seed,work.points,work.n,widthMean,i,d*-1,false).loss;
        if(std::isfinite(plus)&&std::isfinite(minus))work.gradient[i]=work.gradient[i]+axes[axis]*((plus-minus)/(2*p.refineEpsilon));
    }
}
VC_TRACE_HD void refinementLineSearch(Fields f,Params p,Seed seed,Node* nodes,
    double& widthSum,double& widthMean,int& widthCount,RefineScratch& work,Output& output) {
            work.accepted=0;
            auto before=replayPoints(f,p,seed,work.points,work.n,widthMean);
            float largest=0;for(int i=work.first;i<work.n;++i)largest=fmaxf(largest,std::sqrt(work.gradient[i].dot(work.gradient[i])));
            if(before.valid&&largest>0&&std::isfinite(largest))for(int backtrack=0;backtrack<8;++backtrack){
                float step=p.refineStep/largest*std::ldexp(1.f,-backtrack);
                bool boundaryValid=true;
                for(int i=0;i<work.n;++i){work.trial[i]=work.points[i];if(i<work.first)continue;
                    auto& node=nodes[work.ids[i]];if(outside(node.point,p))continue;
                    V displacement=work.points[i]-work.gradient[i]*step-node.planeOrigin;
                    displacement=displacement-node.planeNormal*displacement.dot(node.planeNormal);
                    float length=std::sqrt(displacement.dot(displacement));if(length>p.refineCap)displacement*=p.refineCap/length;
                    work.trial[i]=node.planeOrigin+displacement;
                    boundaryValid &= !outside(work.trial[i],p);
                }
                if(!boundaryValid)continue;
                auto after=replayPoints(f,p,seed,work.trial,work.n,widthMean);
                if(!after.valid||!(after.loss<before.loss))continue;
                double sum=0;bool valid=true;
                for(int i=0;i<work.n;++i){float w=sample(f,p,work.trial[i],seed.direction).width;valid&=w>0&&std::isfinite(w);sum+=w;}
                double newMean=sum/work.n;
                // Do not accept an apparent gain which vanishes when the
                // cumulative width state is updated, or creates cutoff stops.
                auto checked=replayPoints(f,p,seed,work.trial,work.n,newMean);
                auto oldChecked=replayPoints(f,p,seed,work.points,work.n,newMean);
                if(!valid||!checked.valid||!oldChecked.valid||!(checked.loss<oldChecked.loss))continue;
                for(int i=work.first;i<work.n;++i){
                    work.points[i]=work.trial[i];auto& node=nodes[work.ids[i]];node.point=work.points[i];
                    V d=node.point-node.planeOrigin;
                    output.maxPlaneError=fmaxf(output.maxPlaneError,std::abs(d.dot(node.planeNormal)));
                    output.maxMovement=fmaxf(output.maxMovement,std::sqrt(d.dot(d)));
                }
                widthSum=sum;widthCount=work.n;widthMean=newMean;
                State state=initialState(seed);
                for(int i=1;i<work.n;++i){auto& node=nodes[work.ids[i]];
                    state=scoreAtPoint(f,p,state,node.point,unit(node.point-state.point),widthMean,0);
                    node.modulated=state.modulated;node.scoringWidth=widthMean;
                }
                ++output.refinementAccepted;output.refinementGain+=before.loss-after.loss;work.accepted=1;break;
            }
}
// Uniform block call. Gradients run in parallel over movable nodes, and each
// trace/seed owns its state. No locks or Python processing in this path.
__device__ void refinePrefix(Fields f,Params p,Seed seed,Node* nodes,int committed,
                            double& widthSum,double& widthMean,int& widthCount,
                            RefineScratch& work,Output& output) {
    if(!p.refineIterations||committed==0)return;
    if(threadIdx.x==0){
        ++output.refinementCommits;work.n=nodes[committed].depth+1;
        for(int at=committed;at>=0;at=nodes[at].previous){int i=nodes[at].depth;work.ids[i]=at;work.points[i]=nodes[at].point;}
        work.first=p.refineWindow?max(1,work.n-p.refineWindow):1;
    }__syncthreads();
    for(int iteration=0;iteration<p.refineIterations;++iteration){
        for(int i=work.first+threadIdx.x;i<work.n;i+=blockDim.x)
            refinementGradient(f,p,seed,nodes,widthMean,work,i);
        __syncthreads();
        if(threadIdx.x==0)refinementLineSearch(f,p,seed,nodes,widthSum,widthMean,widthCount,work,output);
        __syncthreads();
        if(!work.accepted)break;
    }
}
VC_TRACE_HD State rebuildState(Fields f,Params p,Seed seed,Node* nodes,State old,int committed,double widthMean) {
    int chain[MAX_STEPS+1],n=0;for(int at=old.node;at>0;at=nodes[at].previous)chain[n++]=at;
    State s=initialState(seed);
    for(int i=n-1;i>=0;--i){auto& node=nodes[chain[i]];
        double width=node.depth<=nodes[committed].depth?node.scoringWidth:widthMean;
        s=scoreAtPoint(f,p,s,node.point,unit(node.point-s.point),width,old.original);
        if(!s.valid){s.node=chain[i];return s;}
    }s.node=old.node;return s;
}
__device__ void commitPending(Fields f,Params p,Seed seed,Node* nodes,const int* pending,int n,
    int& observed,int& last,int& stop,int& reason,double& widthSum,double& widthMean,int& widthCount,
    RefineScratch& scratch,Output& output) {
    __shared__ int failed;
    if(threadIdx.x==0){failed=0;if(p.refineIterations)freezePlanes(nodes,pending,n);}
    __syncthreads();
    for(int k=n-1;k>=0;--k){
        if(threadIdx.x==0){
            int at=pending[k];float width=sample(f,p,nodes[at].point,V{1,0,0}).width;
            if(p.refineIterations){State s;s.node=at;s=rebuildState(f,p,seed,nodes,s,observed,widthMean);
                nodes[at].modulated=s.valid?s.modulated:-INFINITY;nodes[at].scoringWidth=widthMean;}
            if(!(width>0)){stop=1;reason=4;last=observed;failed=1;}
            else {
                widthSum+=width;widthMean=widthSum/++widthCount;
                if(nodes[at].modulated<p.cutoff){output.hasRejected=true;output.rejected=nodes[at].point;
                    output.rejectedScore=nodes[at].modulated;output.rejectedWidth=nodes[at].scoringWidth;
                    stop=1;reason=1;last=observed;failed=1;}
                else {observed=at;if(outside(nodes[at].point,p)){stop=1;reason=2;last=observed;failed=2;}}
            }
        }__syncthreads();
        if(failed==1)break;
        refinePrefix(f,p,seed,nodes,observed,widthSum,widthMean,widthCount,scratch,output);
        if(failed)break;
    }
}
// Same multiplicative bound as the native scorer: replace unknown direction
// factors with one, omit nonnegative smoothness. Width/presence math is shared
// with the full sampler; no reduced-precision or approximate angular search.
VC_TRACE_HD float candidateBound(Fields fields,Params p,const State& parent,float u,float v,double widthMean) {
    V forward=unit(parent.current,{1,0,0});auto basis=portable::traceBasis(forward);
    V direction=unit(forward+basis[0]*u+basis[1]*v,forward);
    auto s=sample<true>(fields,p,parent.point+direction*p.step,direction);
    float scale=portable::widthPresenceScale(s.width,widthMean,p.tolerance,p.cosWidth);
    float bound=parent.loss+portable::fiberLocalAlignmentLossInline(s.presence*scale,parent.previous,direction,parent.current,V{},nullptr,true);
    return std::isfinite(bound)?bound:-INFINITY;
}
__device__ int prefer(State* c,int a,int b) {
    return b>=0&&(a<0||better(c[b],c[a]))?b:a;
}
// Uniformly called by the entire block. Three block barriers per winner, with
// warp reductions in between; retain original candidate IDs for exact ties.
__device__ int blockBest(State* c,int best,int* warpBest) {
    for(int d=16;d;d/=2) {
        int other=__shfl_down_sync(0xffffffff,best,d);
        if((threadIdx.x&31)+d<32)best=prefer(c,best,other);
    }
    if(!(threadIdx.x&31))warpBest[threadIdx.x/32]=best;
    __syncthreads();
    if(threadIdx.x<32) {
        best=threadIdx.x<blockDim.x/32?warpBest[threadIdx.x]:-1;
        for(int d=16;d;d/=2) {
            int other=__shfl_down_sync(0xffffffff,best,d);
            if(threadIdx.x+d<32)best=prefer(c,best,other);
        }
        if(threadIdx.x==0)warpBest[0]=best;
    }
    __syncthreads();
    // Every warp must snapshot the winner before the next reduction reuses
    // warpBest[0]. Otherwise a fast warp can overwrite it while another warp
    // is still reading, producing divergent selection/termination decisions.
    const int winner=warpBest[0];
    __syncthreads();
    return winner;
}
__device__ void selectCandidates(State* c,int count,State* selected,int& selectedCount,
    int limit,float prune,bool parallel,int* indices,int* warpBest) {
    if(threadIdx.x==0)selectedCount=0;
    __syncthreads();
    for(int j=0;j<limit;++j) {
        int best=-1;
        if(parallel||threadIdx.x==0)for(int i=parallel?threadIdx.x:0;i<count;i+=parallel?blockDim.x:1) {
            if(!c[i].valid||!std::isfinite(c[i].loss))continue;
            // Winners are strictly ordered by (loss, original ID). This also
            // excludes every previously selected candidate in O(1), instead
            // of rescanning the selected-index list for the top32 stage.
            if(j&&!better(selected[j-1],c[i]))continue;
            bool rejected=false;
            if(prune>0)for(int k=0;k<j;++k) {
                V delta=c[i].point-selected[k].point;
                rejected|=delta.dot(delta)<prune*prune;
            }
            if(!rejected)best=prefer(c,best,i);
        }
        if(parallel)best=blockBest(c,best,warpBest);
        else {if(threadIdx.x==0)warpBest[0]=best;__syncthreads();best=warpBest[0];}
        if(best<0)break;
        if(threadIdx.x==0){indices[j]=best;selected[j]=c[best];selectedCount=j+1;}
        __syncthreads();
    }
}
// Re-evaluate the provisional diversity beam after every expansion: adding
// candidates can change its worst score, so a single top-K threshold is unsafe.
// Keep the immutable parameter block in its caller-owned storage. Passing a
// second by-value copy here produced invalid local-memory reads in optimized
// CUDA 12.8 builds at the second expansion (verified with memcheck on H100).
__device__ void scoreAndSelect(Fields fields,const Params& p,const State* inputs,int inputCount,
    const float2* offsets,double widthMean,State* candidates,State* selected,int& selectedCount,
    int limit,float prune,bool second,bool parallel,bool bounded,bool audit,
    float* bounds,unsigned char* evaluated,int* indices,int* warpBest,Output& output) {
    int count=inputCount*p.dirs;
    for(int i=threadIdx.x;i<count;i+=blockDim.x) {
        auto parent=inputs[i/p.dirs];auto off=offsets[i%p.dirs];
        int original=second?parent.original*p.dirs+i%p.dirs:i;
        if(bounded)bounds[i]=candidateBound(fields,p,parent,off.x,off.y,widthMean);
        evaluated[i]=!bounded||i%p.dirs<8;
        if(evaluated[i])candidates[i]=candidate(fields,p,parent,off.x,off.y,widthMean,original);
        else {candidates[i].valid=false;candidates[i].presenceRejected=false;candidates[i].loss=INFINITY;}
    }
    __syncthreads();
    for(;;) {
        selectCandidates(candidates,count,selected,selectedCount,limit,prune,parallel,indices,warpBest);
        if(!bounded)break;
        float threshold=selectedCount==limit?selected[limit-1].loss:INFINITY;
        int changed=0;
        for(int i=threadIdx.x;i<count;i+=blockDim.x)if(!evaluated[i]&&bounds[i]<=threshold) {
            auto parent=inputs[i/p.dirs];auto off=offsets[i%p.dirs];
            candidates[i]=candidate(fields,p,parent,off.x,off.y,widthMean,second?parent.original*p.dirs+i%p.dirs:i);
            evaluated[i]=1;changed=1;
        }
        if(!__syncthreads_or(changed))break;
    }
    unsigned scored=0,violations=0;
    for(int i=threadIdx.x;i<count;i+=blockDim.x) {
        scored+=evaluated[i];
        if(audit) {
            auto parent=inputs[i/p.dirs];auto off=offsets[i%p.dirs];
            auto full=candidate(fields,p,parent,off.x,off.y,widthMean,second?parent.original*p.dirs+i%p.dirs:i);
            float bound=candidateBound(fields,p,parent,off.x,off.y,widthMean);
            violations+=full.valid&&bound>full.loss;
        }
    }
    atomicAdd(&output.scored,static_cast<unsigned long long>(scored));
    if(audit)atomicAdd(&output.boundViolations,static_cast<unsigned long long>(violations));
    if(threadIdx.x==0) {
        output.candidates+=count;
        if(!selectedCount) {
            // inputs[0] is the last valid lowest-cost parent, also used by CPU.
            int best=-1;
            for(int i=0;i<p.dirs;++i)if(candidates[i].presenceRejected &&
                (best<0||better(candidates[i],candidates[best])))best=i;
            if(best>=0){output.hasRejected=true;output.rejected=candidates[best].point;
                output.rejectedScore=candidates[best].modulated;output.rejectedWidth=widthMean;}
        }
    }
    __syncthreads();
}
__global__ void trace(Fields fields,Params p,const Seed* seeds,const float2* offsets,
    State* allCandidates,Node* allNodes,Output* outputs,int count,bool parallel=false,bool bounded=false,bool audit=false) {
    int id=blockIdx.x;if(id>=count)return;
    State* candidates=allCandidates+size_t(id)*MAX_CAND;
    Node* nodes=allNodes+size_t(id)*MAX_NODES;
    Output& output=outputs[id];
    __shared__ __align__(16) unsigned char beamStorage[sizeof(State)*MAX_BEAM],parentStorage[sizeof(State)*MAX_PARENT];
    State* beam=reinterpret_cast<State*>(beamStorage);State* parents=reinterpret_cast<State*>(parentStorage);
    __shared__ int beamCount,parentCount,nodeCount,observed,stop,last,reason;
    __shared__ double widthSum,widthMean;
    __shared__ int widthCount;
    __shared__ float bounds[MAX_CAND];
    __shared__ unsigned char evaluated[MAX_CAND];
    __shared__ int indices[MAX_PARENT],warpBest[32];
    __shared__ int pending[MAX_STEPS],nPending;
    extern __shared__ __align__(16) unsigned char refinementStorage[];
    auto& refinement=*reinterpret_cast<RefineScratch*>(refinementStorage);
    if(threadIdx.x==0) {
        output.count=0;output.hasRejected=false;output.candidates=0;output.scored=0;output.boundViolations=0;reason=0;stop=0;nodeCount=1;observed=0;last=0;
        output.refinementCommits=output.refinementAccepted=output.refinedFrontiers=0;output.refinementGain=output.maxPlaneError=output.maxMovement=0;
        auto seed=seeds[id];nodes[0]={seed.point,-1,0,1};
        beam[0]={};beam[0].point=seed.point;beam[0].previous=beam[0].current=beam[0].history=unit(seed.direction,{1,0,0});beamCount=1;
        widthMean=sample(fields,p,seed.point,seed.direction).width;widthSum=widthMean;widthCount=1;
        if(!(widthMean>0)){stop=1;reason=4;}
    }
    __syncthreads();
    for(int step=0;step<p.steps&&!stop;step+=2) {
        scoreAndSelect(fields,p,beam,beamCount,offsets,widthMean,candidates,parents,parentCount,
            MAX_PARENT,0,false,parallel,bounded,audit,bounds,evaluated,indices,warpBest,output);
        if(threadIdx.x==0) {
            for(int j=0;j<parentCount;++j) {
                State& chosen=parents[j];
                if(nodeCount>=MAX_NODES){reason=5;stop=1;break;}
                nodes[nodeCount]={chosen.point,chosen.node,nodes[chosen.node].depth+1,chosen.modulated,widthMean};chosen.node=nodeCount++;
            }
            if(!parentCount){stop=1;reason=output.hasRejected?1:3;}else last=parents[0].node;
        }
        __syncthreads();
        if(stop)break;
        scoreAndSelect(fields,p,parents,parentCount,offsets,widthMean,candidates,beam,beamCount,
            MAX_BEAM,p.prune,true,parallel,bounded,audit,bounds,evaluated,indices,warpBest,output);
        if(threadIdx.x==0) {
            for(int j=0;j<beamCount;++j) {
                State& chosen=beam[j];
                if(nodeCount>=MAX_NODES){reason=5;stop=1;break;}
                nodes[nodeCount]={chosen.point,chosen.node,nodes[chosen.node].depth+1,chosen.modulated,widthMean};chosen.node=nodeCount++;
            }
            bool crossing=false;
            for(int k=0;k<beamCount;++k)for(int n=beam[k].node;n!=observed&&n>=0;n=nodes[n].previous)crossing|=outside(nodes[n].point,p);
            if(beamCount)last=beam[0].node;else{reason=output.hasRejected?1:3;stop=1;}
            if(crossing){reason=2;stop=1;}
            if(step+2>=p.steps)stop=1;
            int common=last;
            if(!stop)for(int k=0;k<beamCount;++k) {
                int other=beam[k].node;
                while(nodes[common].depth>nodes[other].depth)common=nodes[common].previous;
                while(nodes[other].depth>nodes[common].depth)other=nodes[other].previous;
                while(common!=other){common=nodes[common].previous;other=nodes[other].previous;}
            }
            nPending=0;
            for(int at=common;at!=observed&&at>=0;at=nodes[at].previous)pending[nPending++]=at;
        }
        __syncthreads();
        commitPending(fields,p,seeds[id],nodes,pending,nPending,observed,last,stop,reason,widthSum,widthMean,widthCount,refinement,output);
        if(p.refineIterations && nPending && !stop){
            if(threadIdx.x==0)++output.refinedFrontiers;
            for(int i=threadIdx.x;i<beamCount;i+=blockDim.x)candidates[i]=rebuildState(fields,p,seeds[id],nodes,beam[i],observed,widthMean);
            __syncthreads();
            int count=beamCount;
            __syncthreads(); // snapshot count before selectCandidates resets beamCount
            selectCandidates(candidates,count,beam,beamCount,MAX_BEAM,p.prune,parallel,indices,warpBest);
            if(threadIdx.x==0){if(beamCount)last=beam[0].node;else{
                last=observed;stop=1;reason=3;
                for(int i=0;i<count;++i)if(candidates[i].presenceRejected){
                    reason=1;output.hasRejected=true;output.rejected=candidates[i].point;
                    output.rejectedScore=candidates[i].modulated;output.rejectedWidth=widthMean;break;
                }
            }}
            __syncthreads();
        }
    }
    if(threadIdx.x==0) {
        // finish(lastValidState): validate any speculative suffix on all exits.
        nPending=0;
        for(int at=last;at!=observed&&at>=0;at=nodes[at].previous)pending[nPending++]=at;
    }__syncthreads();
    commitPending(fields,p,seeds[id],nodes,pending,nPending,observed,last,stop,reason,widthSum,widthMean,widthCount,refinement,output);
    if(threadIdx.x==0) {
        int n=0;for(int at=last;at>=0;at=nodes[at].previous) {
            if(n>MAX_STEPS){reason=5;break;}output.points[n]=nodes[at].point;output.scores[n]=nodes[at].modulated;output.widths[n]=nodes[at].scoringWidth;++n;
        }
        for(int a=0;a<n/2;++a){V tmp=output.points[a];output.points[a]=output.points[n-1-a];output.points[n-1-a]=tmp;float s=output.scores[a];output.scores[a]=output.scores[n-1-a];output.scores[n-1-a]=s;double w=output.widths[a];output.widths[a]=output.widths[n-1-a];output.widths[n-1-a]=w;}
        output.count=n;output.reason=reason;
        if(reason!=1)output.hasRejected=false;
    }
}
__global__ void feasibilityTests(const float* tensors,int* failures) {
    unsigned char bytes[48]={};float widths[8];for(int i=0;i<8;++i)widths[i]=12;
    Params p{};for(int a=0;a<3;++a){p.shape[a]=2;p.origin[a]=0;}p.spacing=1;p.step=.1f;
    Fields f{bytes,widths,tensors,nullptr};
    auto empty=sample(f,p,V{.5f,.5f,.5f},V{1,0,0});
    if(empty.valid || empty.dir.dot(empty.dir)!=0)atomicAdd(failures,1);
    for(int k=0;k<8;++k){bytes[k*6]=100;bytes[k*6+1]=255;bytes[k*6+2]=128;}
    State parent;parent.point={.5f,.5f,.5f};parent.previous=parent.current=parent.history={1,0,0};
    auto control=candidate(f,p,parent,0,0,12,0);
    p.cutoff=double(control.modulated);
    if(!candidate(f,p,parent,0,0,12,0).valid)atomicAdd(failures,1);
    p.cutoff+=1e-8;
    auto rejected=candidate(f,p,parent,0,0,12,0);
    if(rejected.valid || !rejected.presenceRejected)atomicAdd(failures,1);
}

// Exercise the same block-parallel refinement routine on an off-centre tube.
// Scenarios: all/last2/disabled/boundary/variable width. These are deliberately
// small and deterministic, independent of any benchmark cache or GPU model.
__global__ void refinementTests(const float* tensors,int* failures) {
    __shared__ unsigned char bytes[8*8*8*6];
    __shared__ float widths[8*8*8];
    __shared__ __align__(16) unsigned char nodeMemory[5*sizeof(Node)],scratchMemory[sizeof(RefineScratch)],outMemory[sizeof(Output)];
    auto* nodes=reinterpret_cast<Node*>(nodeMemory);auto& work=*reinterpret_cast<RefineScratch*>(scratchMemory);
    auto& output=*reinterpret_cast<Output*>(outMemory);
    __shared__ double sum,mean;__shared__ int count;
    Params p{};for(int a=0;a<3;++a){p.shape[a]=8;p.origin[a]=0;}
    p.spacing=1;p.step=1;p.lo={0,0,0};p.hi={7,7,7};p.cutoff=.05;p.tolerance=.4;
    p.refineIterations=blockIdx.x==2?0:2;p.refineWindow=blockIdx.x==1?2:0;
    p.refineEpsilon=.02f;p.refineStep=.2f;p.refineCap=.6f;
    if(blockIdx.x==3)p.hi.x=5;
    for(int i=threadIdx.x;i<512;i+=blockDim.x){int y=(i/8)%8,z=i/64;
        bytes[6*i]=static_cast<unsigned char>(240*expf(-.15f*((y-4)*(y-4)+(z-3)*(z-3))));
        bytes[6*i+1]=255;bytes[6*i+2]=128;bytes[6*i+3]=255;bytes[6*i+4]=128;bytes[6*i+5]=128;
        widths[i]=blockIdx.x==4?12.f+.2f*y:12.f;
    }
    Seed seed{{1,3.3f,3},{1,0,0}};
    Fields f{bytes,widths,tensors,failures};
    if(threadIdx.x==0){
        output={};count=5;mean=blockIdx.x==4?12.66:12;sum=mean*count;
        for(int i=0;i<5;++i){nodes[i]={};nodes[i].point={float(i+1),3.3f,3};nodes[i].previous=i-1;nodes[i].depth=i;
            nodes[i].planeOrigin=nodes[i].point;nodes[i].planeNormal={1,0,0};nodes[i].planeFrozen=true;nodes[i].scoringWidth=mean;}
    }__syncthreads();
    refinePrefix(f,p,seed,nodes,4,sum,mean,count,work,output);
    if(threadIdx.x==0){
        if(nodes[0].point.x!=1||nodes[0].point.y!=3.3f||nodes[0].point.z!=3)atomicAdd(failures,1);
        for(int i=1;i<5;++i){if(nodes[i].point.x!=i+1||nodes[i].planeOrigin.y!=3.3f||nodes[i].planeNormal.x!=1)atomicAdd(failures,1);}
        if(blockIdx.x==1 && (nodes[1].point.y!=3.3f||nodes[2].point.y!=3.3f))atomicAdd(failures,1);
        if(blockIdx.x==2){if(output.refinementAccepted||nodes[4].point.y!=3.3f)atomicAdd(failures,1);}
        else if(!output.refinementAccepted||!(output.refinementGain>0)||output.maxPlaneError>1e-6||output.maxMovement>p.refineCap+1e-5)atomicAdd(failures,1);
        if(blockIdx.x==3 && nodes[4].point.y!=3.3f)atomicAdd(failures,1);
        V points[5];for(int i=0;i<5;++i)points[i]=nodes[i].point;
        if(!replayPoints(f,p,seed,points,5,mean).valid)atomicAdd(failures,1);
    }
}

__global__ void firstStep(Fields f,Params p,const Seed* seeds,const float2* offsets,State* result,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count*p.dirs)return;
    auto seed=seeds[i/p.dirs];State parent{};parent.point=seed.point;
    parent.previous=parent.current=parent.history=unit(seed.direction,{1,0,0});
    double width=sample(f,p,seed.point,seed.direction).width;
    result[i]=candidate(f,p,parent,offsets[i%p.dirs].x,offsets[i%p.dirs].y,width,i%p.dirs);
}
// Small independent selector tests: ties/permuted IDs, no candidates, partial
// beam, and a diversity chain where a provisional middle winner disappears.
__global__ void selectionTests(int* failures) {
    __shared__ __align__(16) unsigned char cs[sizeof(State)*64],ss[sizeof(State)*32];
    State* c=reinterpret_cast<State*>(cs);State* selected=reinterpret_cast<State*>(ss);
    __shared__ int n,indices[32],warps[32],expected[32],expectedCount;
    __shared__ float losses[64];
    int scenario=blockIdx.x;float prune=scenario==3?1.1f:0;
    for(int i=threadIdx.x;i<64;i+=blockDim.x) {
        c[i]={};c[i].original=63-i;c[i].loss=float(i%5);c[i].point={float(i),0,0};
        c[i].valid=scenario!=1&&(scenario!=2||i<3);losses[i]=c[i].loss;
        if(scenario==3){c[i].loss=float(i);losses[i]=float(i);}
    }
    __syncthreads();
    selectCandidates(c,64,selected,n,8,prune,false,indices,warps);
    if(threadIdx.x==0){expectedCount=n;for(int i=0;i<n;++i)expected[i]=selected[i].original;}
    __syncthreads();
    selectCandidates(c,64,selected,n,8,prune,true,indices,warps);
    if(threadIdx.x==0){if(n!=expectedCount)atomicAdd(failures,1);for(int i=0;i<n;++i)if(selected[i].original!=expected[i])atomicAdd(failures,1);}
    __syncthreads();
    // Initial subset excludes index0 and many lower-ID equal-score contenders.
    // Bounds equal full scores, deliberately including equality at threshold.
    for(int i=threadIdx.x;i<64;i+=blockDim.x)if(i%2==0)c[i].valid=false;
    __syncthreads();
    for(;;) {
        selectCandidates(c,64,selected,n,8,prune,true,indices,warps);
        float threshold=n==8?selected[n-1].loss:INFINITY;int changed=0;
        for(int i=threadIdx.x;i<64;i+=blockDim.x)if(!c[i].valid&&scenario!=1&&(scenario!=2||i<3)&&losses[i]<=threshold){c[i].valid=true;changed=1;}
        if(!__syncthreads_or(changed))break;
    }
    if(threadIdx.x==0){if(n!=expectedCount)atomicAdd(failures,1);for(int i=0;i<n;++i)if(selected[i].original!=expected[i])atomicAdd(failures,1);}
}
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T> T* upload(const std::vector<T>& v){T* p;check(cudaMalloc(&p,v.size()*sizeof(T)));check(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
template<class T> std::vector<T> read(const std::filesystem::path& path){std::ifstream in(path,std::ios::binary|std::ios::ate);if(!in)throw std::runtime_error("missing fixture "+path.string());size_t bytes=in.tellg();if(bytes%sizeof(T))throw std::runtime_error("bad fixture length");std::vector<T> v(bytes/sizeof(T));in.seekg(0);in.read(reinterpret_cast<char*>(v.data()),bytes);if(!in)throw std::runtime_error("short fixture read");return v;}
V vec(const Json& j){return {j[0].get<float>(),j[1].get<float>(),j[2].get<float>()};}
Json refinementReport(const Output& r){return {{"commits",r.refinementCommits},{"accepted",r.refinementAccepted},
    {"frontier_rebuilds",r.refinedFrontiers},{"objective_gain",r.refinementGain},{"max_plane_error",r.maxPlaneError},{"max_movement",r.maxMovement}};}
Params parameters(const Json& j,std::vector<float2>& offsets) {
    if(j["beam"]!=8||j["lookahead"]!=2||j["parent_cap"]!=32||!j["modulated"].get<bool>())throw std::runtime_error("unsupported GPU beam configuration");
    Params p;for(int a=0;a<3;++a){p.origin[a]=j["origin"][a];p.shape[a]=j["shape_xyz"][a];}
    p.spacing=j["spacing"];p.lo=vec(j["bounds"][0]);p.hi=vec(j["bounds"][1]);p.step=j["step"];
    double stepBudget=j["distance"].get<double>()/p.step;
    p.steps=int(std::ceil(stepBudget));if(stepBudget!=p.steps||p.steps>MAX_STEPS||p.steps%2)throw std::runtime_error("unsupported GPU step budget");
    p.prune=j["prune_distance"];p.cutoff=j["cutoff"];p.tolerance=j["width_tolerance"];p.cosWidth=j["width_cosine"];p.maskNormals=j["mask_normals"];
    p.smooth={j["smoothness"],j["normal_smoothness"],j["tangent_smoothness"],j["free_angle"].get<float>()*float(3.14159265358979323846)/180.f};
    p.cumulative=j["cumulative_weight"];p.historySteps=j["cumulative_steps"];
    p.refineIterations=j.value("online_refine_iterations",0);p.refineWindow=j.value("online_refine_window",0);
    p.refineEpsilon=j.value("online_refine_epsilon",.25f);p.refineStep=j.value("online_refine_step",1.f);p.refineCap=j.value("online_refine_cap",8.f);
    if(p.refineIterations<0||p.refineIterations>16||p.refineWindow<0||p.refineWindow>MAX_STEPS||
       !(p.refineEpsilon>0&&std::isfinite(p.refineEpsilon))||!(p.refineStep>0&&std::isfinite(p.refineStep))||!(p.refineCap>0&&std::isfinite(p.refineCap)))
        throw std::runtime_error("invalid online refinement settings");
    for(float w:{p.smooth.isotropicWeight,p.smooth.normalWeight,p.smooth.tangentWeight,p.cumulative})
        if(!std::isfinite(w)||w<0)throw std::runtime_error("bounds require finite nonnegative penalties");
    if(!std::isfinite(p.tolerance)||p.tolerance<0)throw std::runtime_error("invalid width tolerance");
    struct Offset{float u,v,r;int order;};std::vector<Offset> angles;
    double angle=j["angle"],step=j["angle_step"];int radius=int(std::floor(angle/step+1e-6));
    for(int v=-radius;v<=radius;++v)for(int u=-radius;u<=radius;++u)if((u*u+v*v)*step*step<=angle*angle+1e-5)angles.push_back({float(u*step),float(v*step),float((u*u+v*v)*step*step),int(angles.size())});
    std::stable_sort(angles.begin(),angles.end(),[](auto a,auto b){if(a.r!=b.r)return a.r<b.r;if(a.u!=b.u)return a.u<b.u;return a.v<b.v;});
    for(auto a:angles)offsets.push_back({float(std::tan(a.u*3.14159265358979323846/180.)),float(std::tan(a.v*3.14159265358979323846/180.))});
    p.dirs=offsets.size();if(p.dirs!=MAX_DIR)throw std::runtime_error("unsupported GPU angular lattice");
    return p;
}
std::vector<float> tensorTable() {
    std::vector<float> tensors(65536*6);
    for(int x=0;x<256;++x)for(int y=0;y<256;++y) {
        float a=(float(x)-128.f)/127.f,b=(float(y)-128.f)/127.f;
        V axis=vc::core::portable::normalizedFloatOrZero(V{a,b,std::sqrt(std::max(0.f,1.f-a*a-b*b))});
        float values[6]={axis.x*axis.x,axis.x*axis.y,axis.x*axis.z,axis.y*axis.y,axis.y*axis.z,axis.z*axis.z};
        std::copy(values,values+6,tensors.begin()+6*((x<<8)|y));
    }
    return tensors;
}
#ifndef VC_CHUNK_GPU_LIBRARY
void hostRefinementTests() {
    auto tensors=tensorTable();std::vector<unsigned char> bytes(512*6);std::vector<float> widths(512,12);
    for(int i=0;i<512;++i){int y=i/8%8,z=i/64;
        bytes[6*i]=static_cast<unsigned char>(240*std::exp(-.15f*((y-4)*(y-4)+(z-3)*(z-3))));
        bytes[6*i+1]=255;bytes[6*i+2]=128;bytes[6*i+3]=255;bytes[6*i+4]=128;bytes[6*i+5]=128;}
    Fields f{bytes.data(),widths.data(),tensors.data(),nullptr};
    auto require=[](bool value,const char* message){if(!value)throw std::runtime_error(message);};
    for(int scenario=0;scenario<7;++scenario){
        Params p{};for(int a=0;a<3;++a){p.shape[a]=8;p.origin[a]=0;}
        p.spacing=1;p.lo={0,0,0};p.hi={7,7,7};p.cutoff=.05;p.tolerance=.4;p.historySteps=3;
        p.refineEpsilon=.02f;p.refineStep=.2f;p.refineCap=.6f;
        if(scenario==3)p.hi.x=5;
        if(scenario==4)for(int i=0;i<512;++i)widths[i]=12+.2f*(i/8%8);
        if(scenario==5)p.cutoff=.99; // infeasible initial prefix: never modify it
        if(scenario==6)for(int i=0;i<512;++i){int x=i%8,y=i/8%8,z=i/64;
            float center=3.8f+.025f*(x-1)*(x-1),slope=.05f*(x-1),norm=std::sqrt(1+slope*slope);
            bytes[6*i]=static_cast<unsigned char>(240*std::exp(-.15f*((y-center)*(y-center)+(z-3)*(z-3))));
            bytes[6*i+1]=static_cast<unsigned char>(128+127/norm);
            bytes[6*i+2]=static_cast<unsigned char>(128+127*slope/norm);
        }
        Seed seed{{1,3.3f,3},{1,0,0}};Node nodes[5];RefineScratch w{};Output out{};
        double mean=sample(f,p,seed.point,seed.direction).width,sum=5*mean;int count=5;
        w.n=5;w.first=scenario==1?3:1;
        for(int i=0;i<5;++i){nodes[i].point={float(i+1),3.3f,3};nodes[i].previous=i-1;nodes[i].depth=i;
            nodes[i].scoringWidth=mean;w.points[i]=nodes[i].point;w.ids[i]=i;}
        int pending[]={4,3,2,1};freezePlanes(nodes,pending,4);
        float initial=replayPoints(f,p,seed,w.points,5,mean,-1,{},false).loss;
        for(int iteration=0;iteration<(scenario==2?0:2);++iteration){
            for(int i=w.first;i<5;++i)refinementGradient(f,p,seed,nodes,mean,w,i);
            refinementLineSearch(f,p,seed,nodes,sum,mean,count,w,out);
        }
        freezePlanes(nodes,pending,4); // immutable after later calls
        require(nodes[0].point.x==1&&nodes[0].point.y==3.3f,"refinement moved seed");
        for(int i=1;i<5;++i)require(nodes[i].point.x==i+1&&nodes[i].planeOrigin.y==3.3f,"plane origin changed");
        require(out.maxPlaneError<1e-6&&out.maxMovement<=p.refineCap+1e-5,"plane/cap violation");
        if(scenario==1)require(nodes[1].point.y==3.3f&&nodes[2].point.y==3.3f,"window changed fixed prefix");
        if(scenario==2||scenario==5)require(!out.refinementAccepted&&nodes[4].point.y==3.3f,"disabled/infeasible update accepted");
        else {require(out.refinementAccepted>0&&out.refinementGain>0,"off-centre tube did not improve");
            auto final=replayPoints(f,p,seed,w.points,5,mean);require(final.valid&&final.loss<initial,"refinement cost/feasibility regression");}
        if(scenario==3)require(nodes[4].point.y==3.3f,"boundary point moved inside");
        State stale=initialState(seed);stale.node=4;stale.point={5,3.3f,3};stale.loss=1000;
        auto rebuilt=rebuildState(f,p,seed,nodes,stale,4,mean);
        if(scenario==5)require(!rebuilt.valid&&rebuilt.presenceRejected,"cutoff failure lost during rebuild");
        else {require(rebuilt.valid&&rebuilt.point.y==nodes[4].point.y&&rebuilt.loss<1000,"stale frontier state survived");
            require(rebuilt.historySamples>0,"cumulative direction history not rebuilt");}
    }
    std::cout<<"7 shared host refinement tests passed"<<std::endl;
}
int main(int argc,char** argv) try {
    if(argc==2&&std::string(argv[1])=="--host-refinement-tests"){hostRefinementTests();return 0;}
    if(argc==2&&std::string(argv[1])=="--device-tests") {
        // Exercise the actual optimized two-stage search, not just its host
        // scoring functions. Include partial/empty frontiers and repeated
        // winner reductions, with no external dataset requirement.
        constexpr int side=32,count=512;
        Params p{};for(int a=0;a<3;++a){p.shape[a]=side;p.origin[a]=0;}
        p.spacing=1;p.step=.5f;p.steps=16;p.lo={3,3,3};p.hi={28,28,28};
        p.tolerance=.4;p.cutoff=.4;p.prune=.1f;p.historySteps=4;p.cumulative=2;
        std::vector<float2> offsets;
        for(int y=-5;y<=5;++y)for(int x=-5;x<=5;++x)if(x*x+y*y<=25)
            offsets.push_back({float(std::tan(x*.08726646259971647)),float(std::tan(y*.08726646259971647))});
        p.dirs=offsets.size();
        std::vector<unsigned char> bytes(side*side*side*6);
        std::vector<float> widths(side*side*side,12);
        for(int z=0;z<side;++z)for(int y=0;y<side;++y)for(int x=0;x<side;++x){
            const size_t i=(z*side+y)*side+x;
            bytes[6*i]=(x>16&&y>16)?40:240;
            bytes[6*i+1]=255;bytes[6*i+2]=128;
            bytes[6*i+3]=255;bytes[6*i+4]=128;bytes[6*i+5]=128;
        }
        std::vector<Seed> seeds(count);
        for(int i=0;i<count;++i)seeds[i]={{float(8+i%16),float(8+(i/16)%16),12},{1,0,0}};
        Fields f{upload(bytes),upload(widths),upload(tensorTable()),upload(std::vector<int>{0})};
        auto ds=upload(seeds);auto da=upload(offsets);auto failures=upload(std::vector<int>{0});
        State* candidates;Node* nodes;Output* output;
        check(cudaMalloc(&candidates,count*MAX_CAND*sizeof(State)));
        check(cudaMalloc(&nodes,count*MAX_NODES*sizeof(Node)));check(cudaMalloc(&output,count*sizeof(Output)));
        selectionTests<<<4,128>>>(failures);check(cudaGetLastError());
        for(int repeat=0;repeat<3;++repeat){
            trace<<<count,128>>>(f,p,ds,da,candidates,nodes,output,count,true,false,false);
            check(cudaGetLastError());check(cudaDeviceSynchronize());
            std::vector<Output> result(count);check(cudaMemcpy(result.data(),output,count*sizeof(Output),cudaMemcpyDeviceToHost));
            for(const auto& r:result)if(r.count<1||r.count>p.steps+1||r.reason==5)
                throw std::runtime_error("invalid device regression trace");
        }
        int bad=0,field=0;check(cudaMemcpy(&bad,failures,sizeof(int),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(&field,f.error,sizeof(int),cudaMemcpyDeviceToHost));
        cudaFree(candidates);cudaFree(nodes);cudaFree(output);cudaFree(ds);cudaFree(da);cudaFree(failures);
        cudaFree(const_cast<unsigned char*>(f.bytes));cudaFree(const_cast<float*>(f.widths));
        cudaFree(const_cast<float*>(f.tensors));cudaFree(f.error);
        if(bad||field)throw std::runtime_error("device regression failed");
        std::cout<<"CUDA selector and 1536 two-stage trace regressions passed"<<std::endl;return 0;
    }
    if(argc!=3&&argc!=8)throw std::runtime_error("usage: benchmark_chunk_gpu FIXTURE OUTPUT [PARALLEL BOUNDED THREADS AUDIT REPEATS]");
    bool parallel=argc==8&&std::stoi(argv[3]),bounded=argc==8&&std::stoi(argv[4]),audit=argc==8&&std::stoi(argv[6]);
    int threads=argc==8?std::stoi(argv[5]):128,repeats=argc==8?std::stoi(argv[7]):3;
    if((threads!=64&&threads!=128&&threads!=256&&threads!=512)||repeats<1)throw std::runtime_error("invalid launch parameters");
    auto start=std::chrono::steady_clock::now();
    std::filesystem::path folder=argv[1];Json j=Json::parse(std::ifstream(folder/"gpu_input.json"));
    std::vector<float2> offsets;auto p=parameters(j,offsets);
    std::vector<Seed> seeds;for(const auto& s:j["seeds"]) {V pos=vec(s["seed"]),dir=vec(s["direction"]);seeds.push_back({pos,dir});seeds.push_back({pos,dir*-1});}
    auto bytes=read<unsigned char>(folder/"fields.u8");auto widths=read<float>(folder/"width.f32");
    if(widths.size()!=size_t(p.shape[0])*p.shape[1]*p.shape[2]||bytes.size()!=widths.size()*6)throw std::runtime_error("fixture shape mismatch");
    auto tensors=tensorTable();
    auto uploadStart=std::chrono::steady_clock::now();
    Fields fields{upload(bytes),upload(widths),upload(tensors),upload(std::vector<int>{0})};auto deviceSeeds=upload(seeds);auto deviceOffsets=upload(offsets);
    State* candidates;Node* nodes;Output* output;size_t count=seeds.size();
    check(cudaMalloc(&candidates,count*MAX_CAND*sizeof(State)));check(cudaMalloc(&nodes,count*MAX_NODES*sizeof(Node)));check(cudaMalloc(&output,count*sizeof(Output)));
    cudaFuncAttributes attrs;check(cudaFuncGetAttributes(&attrs,trace));
    if(threads>attrs.maxThreadsPerBlock)throw std::runtime_error("thread count exceeds kernel register/resource limit: "+std::to_string(attrs.maxThreadsPerBlock));
    auto testFailures=upload(std::vector<int>{0});
    selectionTests<<<4,threads>>>(testFailures);check(cudaGetLastError());
    feasibilityTests<<<1,1>>>(fields.tensors,testFailures);check(cudaGetLastError());
    refinementTests<<<5,128>>>(fields.tensors,testFailures);check(cudaGetLastError());
    int failed=0;check(cudaMemcpy(&failed,testFailures,sizeof(int),cudaMemcpyDeviceToHost));cudaFree(testFailures);
    if(failed)throw std::runtime_error("synthetic selection/refinement regression failures: "+std::to_string(failed));
    check(cudaDeviceSynchronize());auto ready=std::chrono::steady_clock::now();
    Json timings=Json::array();std::vector<Output> result(count),previous(count);bool repeated=true;
    for(int repeat=0;repeat<repeats;++repeat){auto begin=std::chrono::steady_clock::now();trace<<<count,threads,p.refineIterations?sizeof(RefineScratch):0>>>(fields,p,deviceSeeds,deviceOffsets,candidates,nodes,output,count,parallel,bounded,audit);check(cudaGetLastError());check(cudaDeviceSynchronize());timings.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count());
        check(cudaMemcpy(result.data(),output,count*sizeof(Output),cudaMemcpyDeviceToHost));
        if(repeat)for(size_t i=0;i<count;++i){repeated&=previous[i].count==result[i].count&&previous[i].reason==result[i].reason;for(int k=0;k<std::min(previous[i].count,result[i].count);++k)for(int a=0;a<3;++a)repeated&=previous[i].points[k][a]==result[i].points[k][a];}previous=result;
    }
    auto downloadStart=std::chrono::steady_clock::now();check(cudaMemcpy(result.data(),output,count*sizeof(Output),cudaMemcpyDeviceToHost));
    double download=std::chrono::duration<double>(std::chrono::steady_clock::now()-downloadStart).count();
    // Isolate steady payload transfer from context creation/allocations. These
    // copies are deliberately outside trace timing and reuse existing buffers.
    Json reusableUpload=Json::array();
    for(int k=0;k<3;++k) {
        auto begin=std::chrono::steady_clock::now();
        check(cudaMemcpy(const_cast<unsigned char*>(fields.bytes),bytes.data(),bytes.size(),cudaMemcpyHostToDevice));
        check(cudaMemcpy(const_cast<float*>(fields.widths),widths.data(),widths.size()*sizeof(float),cudaMemcpyHostToDevice));
        check(cudaMemcpy(deviceSeeds,seeds.data(),seeds.size()*sizeof(Seed),cudaMemcpyHostToDevice));
        check(cudaDeviceSynchronize());
        reusableUpload.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count());
    }
    int fieldError=0;check(cudaMemcpy(&fieldError,fields.error,sizeof(int),cudaMemcpyDeviceToHost));
    if(fieldError)throw std::runtime_error("GPU trace accessed outside exported field");
    for(const auto& r:result)if(r.reason==5)throw std::runtime_error("GPU trace exceeded device capacity");
    // Independent host/device first-step check, outside timed trace region.
    firstStep<<<(count*p.dirs+127)/128,128>>>(fields,p,deviceSeeds,deviceOffsets,candidates,count);check(cudaGetLastError());check(cudaDeviceSynchronize());
    std::vector<State> first(count*p.dirs);check(cudaMemcpy(first.data(),candidates,first.size()*sizeof(State),cudaMemcpyDeviceToHost));
    Fields host{bytes.data(),widths.data(),tensors.data(),nullptr};
    size_t validityMismatch=0,lossMismatch=0;double maxLoss=0,maxPosition=0;
    for(size_t i=0;i<first.size();++i){auto seed=seeds[i/p.dirs];State parent{};parent.point=seed.point;parent.previous=parent.current=parent.history=unit(seed.direction,{1,0,0});
        double width=sample(host,p,seed.point,seed.direction).width;
        auto expected=candidate(host,p,parent,offsets[i%p.dirs].x,offsets[i%p.dirs].y,width,i%p.dirs);auto actual=first[i];
        validityMismatch+=expected.valid!=actual.valid;
        if(expected.valid&&actual.valid){lossMismatch+=expected.loss!=actual.loss;maxLoss=std::max(maxLoss,double(std::abs(expected.loss-actual.loss)));for(int a=0;a<3;++a)maxPosition=std::max(maxPosition,double(std::abs(expected.point[a]-actual.point[a])));}
    }
    unsigned long long total=0,scored=0,violations=0;
    Json traces=Json::array();for(const auto& r:result){Json points=Json::array(),scores=Json::array(),widthsJson=Json::array();for(int i=0;i<r.count;++i){points.push_back({r.points[i].x,r.points[i].y,r.points[i].z});scores.push_back(r.scores[i]);widthsJson.push_back(r.widths[i]);}traces.push_back({{"points",points},{"reason",r.reason},{"candidates",r.candidates},{"modulated_scores",scores},{"scoring_width_means",widthsJson}});if(r.hasRejected)traces.back()["cutoff"]={{"point",{r.rejected.x,r.rejected.y,r.rejected.z}},{"score",r.rejectedScore},{"scoring_width_mean",r.rejectedWidth}};total+=r.candidates;scored+=r.scored;violations+=r.boundViolations;}
    cudaDeviceProp prop;check(cudaGetDeviceProperties(&prop,0));
    Json report={{"device",prop.name},{"repeat_exact",repeated},{"first_step_validation",{{"candidate_count",first.size()},{"validity_mismatches",validityMismatch},{"loss_mismatches",lossMismatch},{"max_loss_delta",maxLoss},{"max_position_delta",maxPosition}}},{"fixture",folder.string()},{"trace_seconds",timings},{"upload_alloc_seconds",std::chrono::duration<double>(ready-uploadStart).count()},{"host_prepare_seconds",std::chrono::duration<double>(uploadStart-start).count()},{"download_seconds",download},{"seed_halves",count},{"traces",traces}};
    report["optimization"]={{"parallel",parallel},{"bounded",bounded},{"threads",threads},{"audit",audit},{"candidates",total},{"scored",scored},{"bound_violations",violations}};
    report["kernel_resources"]={{"registers_per_thread",attrs.numRegs},{"shared_bytes",attrs.sharedSizeBytes},{"local_bytes_per_thread",attrs.localSizeBytes},{"max_threads",attrs.maxThreadsPerBlock}};
    report["selection_tests_passed"]=true;
    report["refinement_tests_passed"]=true;
    report["parameters"]=j;
    for(size_t i=0;i<result.size();++i)report["traces"][i]["refinement"]=refinementReport(result[i]);
    report["reusable_upload_seconds"]=reusableUpload;
    report["reusable_upload_bytes"]=bytes.size()+widths.size()*sizeof(float)+seeds.size()*sizeof(Seed);
    std::ofstream out(argv[2]);out<<report.dump(2)<<'\n';if(!out)throw std::runtime_error("report write failed");
    if(!repeated||violations)throw std::runtime_error("GPU repeat/bound audit failed; see report");
    cudaFree(candidates);cudaFree(nodes);cudaFree(output);cudaFree(deviceSeeds);cudaFree(deviceOffsets);cudaFree(fields.error);cudaFree(const_cast<unsigned char*>(fields.bytes));cudaFree(const_cast<float*>(fields.widths));cudaFree(const_cast<float*>(fields.tensors));
    std::cout<<"GPU trace seconds "<<timings<<"; seed halves "<<count<<std::endl;
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
#else
template<class T> struct Buffer {
    T* data=nullptr;size_t capacity=0;
    ~Buffer(){if(data)cudaFree(data);}
    void reserve(size_t n){if(n>capacity){if(data)check(cudaFree(data));data=nullptr;capacity=0;check(cudaMalloc(&data,n*sizeof(T)));capacity=n;}}
    void copy(const T* src,size_t n){reserve(n);if(n)check(cudaMemcpy(data,src,n*sizeof(T),cudaMemcpyHostToDevice));}
};
struct Workspace {
    Buffer<unsigned char> bytes;Buffer<float> widths,tensors;
    Buffer<int> error;Buffer<Seed> seeds;Buffer<float2> offsets;
    Buffer<State> candidates;Buffer<Node> nodes;Buffer<Output> output;
};
// One persistent host thread owns each device and its reusable workspace.
// The caller retains input arrays until this synchronous call returns.
extern "C" const char* vc_chunk_gpu_run(int device,const char* metadata,
    const unsigned char* bytes,size_t byteCount,const float* widths,size_t widthCount) noexcept {
    static thread_local std::string response;
    try {
        check(cudaSetDevice(device));
        static thread_local Workspace w;
        auto j=Json::parse(metadata);std::vector<float2> offsets;auto p=parameters(j,offsets);
        if(widthCount!=size_t(p.shape[0])*p.shape[1]*p.shape[2]||byteCount!=6*widthCount)
            throw std::runtime_error("GPU field shape mismatch");
        auto begin=std::chrono::steady_clock::now();
        w.bytes.copy(bytes,byteCount);w.widths.copy(widths,widthCount);
        if(!w.tensors.capacity){auto t=tensorTable();w.tensors.copy(t.data(),t.size());}
        int zero=0;w.error.copy(&zero,1);w.offsets.copy(offsets.data(),offsets.size());
        std::vector<Seed> seeds;
        for(const auto& s:j["seeds"]){auto point=vec(s["seed"]),dir=vec(s["direction"]);seeds.push_back({point,dir});seeds.push_back({point,dir*-1});}
        constexpr size_t batch=1024;
        const size_t capacity=std::min(batch,seeds.size());
        w.candidates.reserve(capacity*MAX_CAND);w.nodes.reserve(capacity*MAX_NODES);w.output.reserve(capacity);
        Fields fields{w.bytes.data,w.widths.data,w.tensors.data,w.error.data};
        Json traces=Json::array();double kernelSeconds=0;
        for(size_t start=0;start<seeds.size();start+=batch){
            const size_t count=std::min(batch,seeds.size()-start);w.seeds.copy(seeds.data()+start,count);
            auto k=std::chrono::steady_clock::now();
            trace<<<count,128,p.refineIterations?sizeof(RefineScratch):0>>>(fields,p,w.seeds.data,w.offsets.data,w.candidates.data,w.nodes.data,w.output.data,count,true,false,false);
            check(cudaGetLastError());check(cudaDeviceSynchronize());
            kernelSeconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-k).count();
            std::vector<Output> result(count);check(cudaMemcpy(result.data(),w.output.data,count*sizeof(Output),cudaMemcpyDeviceToHost));
            for(const auto& r:result){
                if(r.reason==5)throw std::runtime_error("GPU trace capacity exceeded");
                Json points=Json::array();for(int i=0;i<r.count;++i)points.push_back({r.points[i].x,r.points[i].y,r.points[i].z});
                traces.push_back({{"points",points},{"reason",r.reason},{"refinement",refinementReport(r)}});
            }
        }
        int error=0;check(cudaMemcpy(&error,w.error.data,sizeof(int),cudaMemcpyDeviceToHost));
        if(error)throw std::runtime_error("GPU trace sampled outside prepared field");
        response=Json{{"traces",traces},{"device",device},{"kernel_seconds",kernelSeconds},
            {"seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()}}.dump();
    }catch(const std::exception& e){response=Json{{"error",e.what()}}.dump();}
    return response.c_str();
}
#endif
