// Host-orchestrated GPU regression tests, only in the standalone benchmark.
__global__ void rollingSelectionTest(int* failed) {
    __shared__ __align__(16) unsigned char cm[sizeof(State)*4],sm[sizeof(State)*4];
    auto* c=reinterpret_cast<State*>(cm);auto* s=reinterpret_cast<State*>(sm);
    __shared__ int count,indices[4],warps[32];
    if(threadIdx.x==0){for(int i=0;i<4;++i){c[i]={};c[i].loss=i+1;c[i].original=i;c[i].node=i<2?0:i-1;}}
    __syncthreads();selectCandidates(c,4,s,count,4,0,true,indices,warps,true);
    if(threadIdx.x==0)if(count!=3||s[0].node!=0||s[1].node!=1||s[2].node!=2)atomicAdd(failed,1);
}
void rollingTests() {
    Params p{};for(int a=0;a<3;++a)p.shape[a]=8;
    p.spacing=1;p.step=.5f;p.lo={0,0,0};p.hi={6.25f,7,7};p.dirs=3;p.steps=12;
    p.rolling=p.analyticRefinement=true;p.cutoff=.1;p.tolerance=.8;p.historySteps=4;
    p.refineStep=.1;p.refineCap=.5;p.refineWindow=2;
    std::vector<unsigned char> bytes(512*6);std::vector<float> widths(512);
    for(int i=0;i<512;++i){int y=i/8%8,z=i/64;
        bytes[i*6]=static_cast<unsigned char>(240*std::exp(-.1f*((y-4)*(y-4)+(z-3)*(z-3))));
        bytes[i*6+1]=255;bytes[i*6+2]=128;bytes[i*6+3]=255;bytes[i*6+4]=128;bytes[i*6+5]=128;
        widths[i]=10.f+y;
    }
    auto table=tensorTable();Fields f{upload(bytes),upload(widths),upload(table),upload(std::vector<int>{0})};
    auto* offsets=upload(std::vector<float2>{{0,0},{.2f,0},{-.2f,0}});
    Seed seed{{1,3.3f,3},{1,0,0}};auto* ds=upload(std::vector<Seed>{seed});
    State* candidates;Node* nodes;Output* output;
    check(cudaMalloc(&candidates,MAX_CAND*sizeof(State)));check(cudaMalloc(&nodes,MAX_NODES*sizeof(Node)));check(cudaMalloc(&output,sizeof(Output)));
    RollingWorkspace workspace;
    auto run=[&](Params q,Seed s){check(cudaMemcpy(ds,&s,sizeof(s),cudaMemcpyHostToDevice));configureRefinementKernel(q);
        launchTrace(f,q,ds,offsets,candidates,nodes,output,1,128,true,false,false,workspace);check(cudaDeviceSynchronize());
        Output out;check(cudaMemcpy(&out,output,sizeof(out),cudaMemcpyDeviceToHost));int error=0;check(cudaMemcpy(&error,f.error,sizeof(error),cudaMemcpyDeviceToHost));
        if(error)throw std::runtime_error("rolling test sampled outside fields");return out;};
    auto require=[](bool v,const char* message){if(!v)throw std::runtime_error(message);};
    rollingSelectionTest<<<1,128>>>(f.error);check(cudaGetLastError());
    Fields host{bytes.data(),widths.data(),table.data(),nullptr};
    auto a=run(p,seed);auto b=run(p,seed);
    require(a.count>2&&a.reason==2&&a.count==b.count,"rolling boundary progression failed");
    double sum=sample(host,p,seed.point,seed.direction).width;
    for(int i=1;i<a.count;++i){require(a.points[i].x==b.points[i].x&&a.points[i].y==b.points[i].y&&a.points[i].z==b.points[i].z,"rolling repeat differs");
        require(std::abs(a.widths[i]-sum/i)<1e-5,"speculative width leaked into committed mean");sum+=sample(host,p,a.points[i],seed.direction).width;}
    auto q=p;q.steps=3;auto odd=run(q,seed);require(odd.count==4&&odd.reason==0,"rolling odd final horizon failed");
    q=p;q.steps=2;auto edge=run(q,Seed{{6,3.3f,3},{1,0,0}});
    require(edge.count==2&&edge.reason==2,"committed boundary requires extra lookahead");
    q=p;q.cutoff=.999;auto failed=run(q,seed);require(failed.count==1&&failed.reason==1,"all-invalid candidate stop failed");
    q=p;q.steps=1;q.refineIterations=1;auto refined=run(q,seed);
    require(refined.count==2&&refined.reason==0&&refined.refinementCommits==1,"per-step refinement failed");
    // Inspect both private banks: all sibling paths advanced, source bank untouched.
    std::vector<RollingBranch> branches(MAX_BEAM*2);std::vector<Node> paths(ROLLING_NODES);
    check(cudaMemcpy(branches.data(),workspace.branches,branches.size()*sizeof(RollingBranch),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(paths.data(),nodes,paths.size()*sizeof(Node),cudaMemcpyDeviceToHost));
    require(paths[0].point.y==seed.point.y&&branches[0].state.widthCount==1,"parent bank mutated");
    int siblings=0;bool differentWidths=false;double firstWidth=0;
    for(int j=0;j<MAX_BEAM;++j){auto& branch=branches[MAX_BEAM+j];if(!branch.present)continue;++siblings;
        auto& point=paths[MAX_BEAM*ROLLING_PATH+j*ROLLING_PATH+1];
        require(branch.state.widthCount==2&&branch.stats.refinementCommits==1,"sibling didn't independently commit/refine");
        require(point.planeFrozen&&branch.stats.maxPlaneError<1e-5,"rolling plane changed");
        if(j==0)firstWidth=branch.state.widthSum;else differentWidths|=branch.state.widthSum!=firstWidth;
        double expected=sample(host,q,seed.point,seed.direction).width+sample(host,q,point.point,seed.direction).width;
        require(std::abs(branch.state.widthSum-expected)<1e-5,"refined branch width not its own path sum");
    }
    require(siblings>=2&&differentWidths,"forked beams didn't retain independent width state");
    // Synthetic finish state: invalid top-ranked refinement must not stop a live sibling.
    branches[MAX_BEAM].refinementRejected=true;branches[MAX_BEAM].reason=1;
    branches[MAX_BEAM+1].present=true;branches[MAX_BEAM+1].refinementRejected=false;branches[MAX_BEAM+1].reason=-1;
    Output active{};active.reason=-1;check(cudaMemcpy(output,&active,sizeof(active),cudaMemcpyHostToDevice));
    check(cudaMemcpy(workspace.branches,branches.data(),branches.size()*sizeof(RollingBranch),cudaMemcpyHostToDevice));
    rollingFinish<<<1,128>>>(workspace.branches+MAX_BEAM,nodes+MAX_BEAM*ROLLING_PATH,output,1);check(cudaGetLastError());
    check(cudaMemcpy(&active,output,sizeof(active),cudaMemcpyDeviceToHost));require(active.reason==-1,"rejected winner terminated live sibling");
    // Several rolling advances refine every surviving lineage, without waiting
    // for common ancestry. Width count and immutable plane origins survive forks.
    q=p;q.steps=5;q.refineIterations=1;auto multi=run(q,seed);
    require(multi.count==6&&multi.refinementCommits==5,"rolling refinement delayed until common ancestry");
    check(cudaMemcpy(branches.data(),workspace.branches,branches.size()*sizeof(RollingBranch),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(paths.data(),nodes,paths.size()*sizeof(Node),cudaMemcpyDeviceToHost));
    for(int j=0;j<MAX_BEAM;++j){auto& branch=branches[MAX_BEAM+j];if(!branch.present)continue;
        require(branch.state.widthCount==6,"multi-step width count wrong");
        double expected=0;for(int i=0;i<=branch.depth;++i){auto& node=paths[MAX_BEAM*ROLLING_PATH+j*ROLLING_PATH+i];
            expected+=sample(host,q,node.point,seed.direction).width;
            if(i)require(node.planeFrozen&&std::abs((node.point-node.planeOrigin).dot(node.planeNormal))<1e-5,"multi-step plane drift");}
        require(std::abs(branch.state.widthSum-expected)<2e-5,"multi-step branch width sum wrong");
    }
    // All initial second-step lookaheads fail; moving the first point toward
    // stronger presence must allow the next advance to be retried successfully.
    for(int i=0;i<512;++i){int x=i%8,y=i/8%8;bytes[i*6]=x<3?(y<=2?80:240):(y<=2?0:128);widths[i]=12;}
    check(cudaMemcpy(const_cast<unsigned char*>(f.bytes),bytes.data(),bytes.size(),cudaMemcpyHostToDevice));
    check(cudaMemcpy(const_cast<float*>(f.widths),widths.data(),widths.size()*sizeof(float),cudaMemcpyHostToDevice));
    q=p;q.dirs=1;q.step=1;q.steps=3;q.cutoff=.4;q.refineStep=.4;q.refineCap=.5;q.hi.x=6;q.smooth={0,0,0,0};
    Seed rescue{{1,2.7f,3},{1,0,0}};
    auto stopped=run(q,rescue);require(stopped.count==2&&stopped.reason==1,"fallback fixture doesn't fail initial lookahead");
    q.refineIterations=1;auto rescued=run(q,rescue);
    std::cout<<"Rescue fixture before/after counts "<<stopped.count<<"/"<<rescued.count<<" accepted "<<rescued.refinementAccepted
        <<" last y "<<rescued.points[rescued.count-1].y<<" rejected score "<<rescued.rejectedScore<<std::endl;
    require(rescued.count>stopped.count,"refinement-rescued lookahead wasn't retried");
    cudaFree(candidates);cudaFree(nodes);cudaFree(output);cudaFree(ds);cudaFree(offsets);
    cudaFree(const_cast<unsigned char*>(f.bytes));cudaFree(const_cast<float*>(f.widths));cudaFree(const_cast<float*>(f.tensors));cudaFree(f.error);
    std::cout<<"Rolling GPU: progression, widths, forks, planes, cutoff, boundary, odd horizon, rejected-winner, unique-parent, rescued-lookahead and repeat checks passed"<<std::endl;
}
