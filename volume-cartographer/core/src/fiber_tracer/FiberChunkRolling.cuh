// Rolling two-step horizon, advancing one private committed point per beam.
// Search is one CTA/seed-half; refinement is one CTA/(seed-half,beam).
// Ping-pong node banks are immutable sources until all children have copied.
struct RollingBranch {
    State state;
    RefinementStats stats;
    int depth=0,reason=-1;
    bool present=false,hasRejected=false,refinementRejected=false;
    V rejected;
    float rejectedScore=0;
    double rejectedWidth=0;
};
constexpr int ROLLING_PATH=MAX_STEPS+1;
constexpr int ROLLING_NODES=2*MAX_BEAM*ROLLING_PATH;
static_assert(ROLLING_NODES<=MAX_NODES);

VC_TRACE_HD void rollingExport(const RollingBranch& b,const Node* path,Output& out,int reason) {
    static_cast<RefinementStats&>(out)=b.stats;
    out.count=b.depth+1;out.reason=reason;out.hasRejected=b.hasRejected&&reason==1;
    out.rejected=b.rejected;out.rejectedScore=b.rejectedScore;out.rejectedWidth=b.rejectedWidth;
    for(int i=0;i<=b.depth;++i){out.points[i]=path[i].point;out.scores[i]=path[i].modulated;out.widths[i]=path[i].scoringWidth;}
}

__global__ void rollingInit(Fields f,Params p,const Seed* seeds,RollingBranch* branches,Node* nodes,Output* outputs,int count) {
    int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=count)return;
    auto seed=seeds[id];auto& b=branches[id*MAX_BEAM];b={};b.present=true;b.state=initialState(seed);
    b.state.widthSum=sample(f,p,seed.point,seed.direction).width;b.state.widthCount=1;
    for(int j=1;j<MAX_BEAM;++j)branches[id*MAX_BEAM+j].present=false;
    auto& n=nodes[id*MAX_BEAM*ROLLING_PATH];n={};n.point=seed.point;n.previous=-1;n.scoringWidth=b.state.widthSum;
    auto& out=outputs[id];out={};out.reason=-1;
    if(!(b.state.widthSum>0&&std::isfinite(b.state.widthSum)))rollingExport(b,&n,out,4);
}

__global__ void rollingSearch(Fields f,Params p,const Seed* seeds,const float2* offsets,
    const RollingBranch* before,RollingBranch* after,const Node* oldNodes,Node* newNodes,
    State* allCandidates,Output* outputs,int count,int step) {
    int id=blockIdx.x;if(id>=count||outputs[id].reason>=0)return;
    auto& output=outputs[id];State* candidates=allCandidates+size_t(id)*MAX_CAND;
    before+=id*MAX_BEAM;after+=id*MAX_BEAM;
    oldNodes+=size_t(id)*MAX_BEAM*ROLLING_PATH;newNodes+=size_t(id)*MAX_BEAM*ROLLING_PATH;
    __shared__ __align__(16) unsigned char inMemory[sizeof(State)*MAX_BEAM],parentMemory[sizeof(State)*MAX_PARENT],selectedMemory[sizeof(State)*MAX_BEAM];
    auto* inputs=reinterpret_cast<State*>(inMemory);auto* parents=reinterpret_cast<State*>(parentMemory);auto* selected=reinterpret_cast<State*>(selectedMemory);
    __shared__ int nInputs,nParents,nSelected,bestInput,indices[MAX_PARENT],warps[32],fallback;
    __shared__ float bounds[MAX_CAND];__shared__ unsigned char evaluated[MAX_CAND];
    if(threadIdx.x==0){
        nInputs=0;bestInput=-1;
        for(int j=0;j<MAX_BEAM;++j){after[j].present=false;
            if(!before[j].present||before[j].reason>=0)continue;
            inputs[nInputs]=before[j].state;inputs[nInputs].node=j;inputs[nInputs].source=j;inputs[nInputs].original=nInputs;
            if(bestInput<0||better(inputs[nInputs],inputs[bestInput]))bestInput=nInputs;
            ++nInputs;
        }
    }__syncthreads();
    // Terminal branches never compete by raw loss with deeper active branches.
    // rollingFinish already exports a winning terminal, so at least one active remains.
    if(!nInputs){if(threadIdx.x==0)output.reason=5;return;}
    scoreAndSelect(f,p,inputs,nInputs,offsets,0,candidates,parents,nParents,
        MAX_PARENT,0,false,true,false,false,bounds,evaluated,indices,warps,output);
    if(!nParents){
        if(threadIdx.x==0){
            int source=inputs[bestInput].source;RollingBranch b=before[source];b.hasRejected=false;
            int rejected=-1;
            for(int i=bestInput*p.dirs;i<(bestInput+1)*p.dirs;++i)if(candidates[i].presenceRejected&&(rejected<0||better(candidates[i],candidates[rejected])))rejected=i;
            if(rejected>=0){b.hasRejected=true;b.rejected=candidates[rejected].point;b.rejectedScore=candidates[rejected].modulated;b.rejectedWidth=inputs[bestInput].widthSum/inputs[bestInput].widthCount;}
            rollingExport(b,oldNodes+source*ROLLING_PATH,output,b.hasRejected?1:3);
        }return;
    }
    // Second-step exploration. Boundary/final-budget first steps are complete:
    // do not sample a needless second point beyond the required field halo.
    for(int i=threadIdx.x;i<nParents*p.dirs;i+=blockDim.x){
        int parent=i/p.dirs;State first=parents[parent];first.node=parent;
        if(outside(first.point,p)||step+1>=p.steps){
            candidates[i]=first;candidates[i].original=parents[parent].original*p.dirs+i%p.dirs;
            candidates[i].valid=i%p.dirs==0;
            // Missing boundary lookahead repeats the last incremental cost for
            // ranking only, rather than winning for having one fewer term.
            // At the final budget ALL branches use the same one-step horizon.
            if(step+1<p.steps){float base=before[first.source].state.loss;
                candidates[i].loss=base+2*(first.loss-base);}
        }else candidates[i]=candidate(f,p,first,offsets[i%p.dirs].x,offsets[i%p.dirs].y,0,parents[parent].original*p.dirs+i%p.dirs);
    }__syncthreads();
    if(threadIdx.x==0){output.candidates+=nParents*p.dirs;output.scored+=nParents*p.dirs;}
    selectCandidates(candidates,nParents*p.dirs,selected,nSelected,MAX_BEAM,p.prune,true,indices,warps,true);
    if(threadIdx.x==0)fallback=nSelected==0;
    __syncthreads();
    if(fallback){
        // All lookaheads failed: retain a valid first point and report the next
        // failed advance, instead of throwing away this last valid step.
        selectCandidates(parents,nParents,selected,nSelected,MAX_BEAM,p.prune,true,indices,warps);
    }
    if(threadIdx.x==0){
        for(int j=0;j<nSelected;++j){
            int parent=fallback?indices[j]:selected[j].node;auto chosen=parents[parent];
            auto& b=after[j];b=before[chosen.source];b.present=true;b.hasRejected=false;b.refinementRejected=false;b.depth=step+1;b.state=chosen;b.state.node=b.depth;
            b.reason=outside(chosen.point,p)?2:(step+1>=p.steps?0:-1);
            // Failed lookahead isn't definitive: refinement can rescue the
            // next advance. Keep this first point active and try again.
        }
    }__syncthreads();
    for(int i=threadIdx.x;i<nSelected*ROLLING_PATH;i+=blockDim.x){
        int j=i/ROLLING_PATH,k=i%ROLLING_PATH;auto& b=after[j];if(k>b.depth)continue;
        Node& node=newNodes[i];int source=b.state.source;
        if(k<b.depth)node=oldNodes[source*ROLLING_PATH+k];
        else {auto& parent=before[source];node={};node.point=b.state.point;node.previous=k-1;node.depth=k;
            node.modulated=b.state.modulated;node.scoringWidth=parent.state.widthSum/parent.state.widthCount;
            node.history=b.state.history;node.prefixLoss=b.state.loss;
            node.planeOrigin=node.point;node.planeNormal=unit(node.point-parent.state.point);node.planeFrozen=true;
        }
    }
}

__global__ void rollingRefine(Fields f,Params p,const Seed* seeds,RollingBranch* branches,Node* nodes,const Output* outputs,int count) {
    int id=blockIdx.x/MAX_BEAM,j=blockIdx.x%MAX_BEAM;if(id>=count||outputs[id].reason>=0)return;
    auto& b=branches[id*MAX_BEAM+j];if(!b.present)return;
    Node* path=nodes+(size_t(id)*MAX_BEAM+j)*ROLLING_PATH;
    extern __shared__ __align__(16) unsigned char scratchMemory[];
    auto& scratch=*reinterpret_cast<RefineScratch*>(scratchMemory);
    __shared__ double sum,mean;__shared__ int n;
    if(threadIdx.x==0){sum=b.state.widthSum;n=b.state.widthCount;mean=sum/n;}
    __syncthreads();
    refinePrefix(f,p,seeds[id],path,b.depth,sum,mean,n,scratch,b.stats);
    if(threadIdx.x==0){
        auto& last=path[b.depth];b.state.point=last.point;b.state.previous=unit(last.point-path[b.depth-1].point);
        b.state.current=sample(f,p,last.point,b.state.previous).dir;b.state.history=last.history;
        b.state.historySamples=std::min(b.depth,std::max(1,p.historySteps));b.state.loss=last.prefixLoss;b.state.modulated=last.modulated;
        b.state.widthSum=sum;b.state.widthCount=n;
        if(p.cutoff>0&&!(last.modulated>=p.cutoff)){
            b.refinementRejected=true;
            b.reason=1;b.hasRejected=true;b.rejected=last.point;b.rejectedScore=last.modulated;b.rejectedWidth=last.scoringWidth;
            --b.depth; // Failed committed step is not emitted as valid geometry.
        }
    }
}

__global__ void rollingFinish(const RollingBranch* branches,const Node* nodes,Output* outputs,int count) {
    int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=count||outputs[id].reason>=0)return;
    // Selection order is the lookahead ordering (before refinement), not a
    // comparison of unequal-length terminal and active prefix losses.
    const auto* choices=branches+id*MAX_BEAM;int best=-1,fallback=-1;
    for(int j=0;j<MAX_BEAM;++j)if(choices[j].present){
        if(fallback<0)fallback=j;
        if(!choices[j].refinementRejected){best=j;break;}
    }
    if(best<0)best=fallback;
    if(best<0){outputs[id].reason=5;return;}
    const auto& winner=choices[best];
    if(winner.reason>=0)rollingExport(winner,nodes+(size_t(id)*MAX_BEAM+best)*ROLLING_PATH,outputs[id],winner.reason);
}
