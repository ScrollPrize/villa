// Included by the CUDA tracer after its shared field/path types. Connectivity
// and planes are immutable. Only local terms and a small direction recurrence
// depend on the movable vertices; no fixed-prefix field samples are replayed.
using LocalDual=differential::Scalar<12>;
template<class S> using LocalVector=differential::Vector<S>;
template<class S> VC_TRACE_HD S liftScalar(float value,V gradient,int offset) {
    S s(value);
    if constexpr(!std::is_same_v<S,float>)if(offset>=0)for(int a=0;a<3;++a)s.gradient[offset+a]=gradient[a];
    return s;
}
template<class S> VC_TRACE_HD LocalVector<S> liftVector(V value,const V* jac,int offset) {
    LocalVector<S> v;
    for(int a=0;a<3;++a){V g{};if(jac)for(int b=0;b<3;++b)g[b]=jac[b][a];v[a]=liftScalar<S>(value[a],g,offset);}
    return v;
}
template<class S> VC_TRACE_HD LocalVector<S> variablePoint(V point,int offset) {
    V jac[3]={{1,0,0},{0,1,0},{0,0,1}};return liftVector<S>(point,jac,offset);
}
template<class S> VC_TRACE_HD S widthScaleDifferential(float width,double mean,const Params& p,V jac) {
    float v=portable::widthPresenceScale(width,mean,p.tolerance,p.cosWidth),derivative=0;
    if(width>0&&p.tolerance>0&&mean>0){double signedDistance=(width/mean-1)/p.tolerance,dist=std::abs(signedDistance);
        if(dist<1)derivative=float((p.cosWidth?-1.5707963267948966*std::sin(1.5707963267948966*dist):-1.)*(signedDistance>0?1:signedDistance<0?-1:0)/(mean*p.tolerance));
    }
    return liftScalar<S>(v,jac*derivative,0);
}
template<bool Gradient> VC_TRACE_HD auto directTerm(Fields f,const Params& p,Seed seed,const V* points,
    int i,V incomingHistory,double mean,float& width,float& modulated,bool& valid) {
    using S=std::conditional_t<Gradient,LocalDual,float>;
    using Q=LocalVector<S>;
    if(!inField(p,points[i])||!inField(p,points[i-1])||(points[i]-points[i-1]).dot(points[i]-points[i-1])<1e-12f){valid=false;return S(INFINITY);}
    Q current=variablePoint<S>(points[i],0),previous=variablePoint<S>(points[i-1],3);
    Q dir=portable::normalizeFiberLocalOrZero(current-previous),prevdir;
    if(i==1)prevdir=liftVector<S>(unit(seed.direction),nullptr,-1);
    else prevdir=portable::normalizeFiberLocalOrZero(previous-variablePoint<S>(points[i-2],6));
    V rawdir=unit(points[i]-points[i-1]);SampleJacobian jac{},prevjac{};
    auto s=sample<false,Gradient>(f,p,points[i],rawdir,&jac);
    Sample prev;
    if(i==1){prev.dir=unit(seed.direction);prev.valid=true;}
    else prev=sample<false,Gradient>(f,p,points[i-1],unit(points[i-1]-points[i-2]),&prevjac);
    Q predicted=liftVector<S>(s.dir,jac.dir,0),oldPrediction=liftVector<S>(prev.dir,prevjac.dir,i==1?-1:3);
    Q normal=liftVector<S>(s.normal,jac.normal,0);
    S presence=liftScalar<S>(s.presence,jac.presence,0)*widthScaleDifferential<S>(s.width,mean,p,jac.width);
    S alignment=portable::fiberLocalAlignmentLossInline(presence,prevdir,dir,oldPrediction,predicted);
    auto smooth=portable::fiberLocalSmoothnessCostInline(prevdir,dir,normal,s.normal.dot(s.normal)>1e-12f,p.smooth);
    S cumulative=0;
    if(p.cumulative>0&&s.normal.dot(s.normal)>1e-12f){
        Q history=variablePoint<S>(incomingHistory,9);
        Q h=portable::normalizeFiberLocalOrZero(history-normal*portable::clampFiberLocalUnit(history.dot(normal)));
        Q d=portable::normalizeFiberLocalOrZero(dir-normal*portable::clampFiberLocalUnit(dir.dot(normal)));
        if(h.dot(h)>1e-12f&&d.dot(d)>1e-12f)cumulative=p.cumulative*portable::fiberLocalExcessAngleSquared(portable::fiberLocalAngleBetweenUnit(h,d),p.smooth.freeAngleRadians);
    }
    if constexpr(Gradient)modulated=1-alignment.value;else modulated=1-alignment;
    width=s.width;valid=s.valid&&prev.valid&&width>0&&std::isfinite(width)&&(!(p.cutoff>0)||modulated>=p.cutoff);
    return alignment+smooth.total()+cumulative;
}
VC_TRACE_HD V normalizedAdjoint(V input,V gradient) {
    float length=std::sqrt(input.dot(input));if(!(length>1e-6f))return {};
    V axis=input/length;return (gradient-axis*axis.dot(gradient))/length;
}
VC_TRACE_HD void directHistories(const Params& p,Seed seed,Node* nodes,RefineScratch& w,const V* points) {
    w.histories[w.first-1]=w.first==1?unit(seed.direction):nodes[w.ids[w.first-1]].history;
    for(int i=w.first;i<w.n;++i){V d=unit(points[i]-points[i-1]);
        w.histories[i]=i==1||p.historySteps<=1?d:unit(d+unit(w.histories[i-1])*float(std::clamp(i-1,1,p.historySteps-1)),d);
    }
}
VC_TRACE_HD void directAdjoint(const Params& p,Node* nodes,RefineScratch& w) {
            V adj{};w.before=0;w.largest=0;w.accepted=0;w.proposal=true;
            for(int i=w.n-1;i>=w.first;--i){
                V step=w.points[i]-w.points[i-1],d=unit(step),previous=w.histories[i-1];
                bool first=i==1||p.historySteps<=1;float weight=first?0:float(std::clamp(i-1,1,p.historySteps-1));
                V a=first?adj:normalizedAdjoint(d+unit(previous)*weight,adj);
                V dg=normalizedAdjoint(step,a);w.gradient[i]=w.gradient[i]+dg;
                if(i-1>=w.first)w.gradient[i-1]=w.gradient[i-1]-dg;
                adj=w.historyGradient[i]+(first?V{}:normalizedAdjoint(previous,a*weight));
                w.proposal&=w.valid[i]!=0;
            }
            for(int i=w.first;i<w.n;++i){auto& n=nodes[w.ids[i]];
                // Same reduction order as trial costs: reject rounded no-ops.
                w.before+=w.losses[i];
                w.gradient[i]=w.gradient[i]-n.planeNormal*w.gradient[i].dot(n.planeNormal);
                if(outside(n.point,p))w.gradient[i]={};
                w.proposal&=std::isfinite(w.gradient[i].x)&&std::isfinite(w.gradient[i].y)&&std::isfinite(w.gradient[i].z);
                w.largest=fmaxf(w.largest,std::sqrt(w.gradient[i].dot(w.gradient[i])));
            }
            w.proposal&=std::isfinite(w.before)&&std::isfinite(w.largest)&&w.largest>0;
}
// Parallel points, analytic local gradients, one cheap recurrence adjoint.
// Line-search candidates share the same fields; no CPU work or kernel launch.
#ifndef VC_CHUNK_GPU_LIBRARY
// Independent value check: use the actual beam scorer, not directTerm, with
// the same frozen prefix state and width reference as the optimization.
VC_TRACE_HD float auditBeamObjective(Fields f,const Params& p,Seed seed,Node* nodes,
    const RefineScratch& w,const V* points,double width) {
    State state=initialState(seed);int previous=w.first-1;
    if(previous>0){state.point=points[previous];state.previous=unit(points[previous]-points[previous-1]);
        state.current=sample(f,p,state.point,state.previous).dir;state.history=nodes[w.ids[previous]].history;
        state.historySamples=std::min(previous,std::max(1,p.historySteps));}
    float total=0;
    for(int i=w.first;i<w.n;++i){state.loss=0;
        state=scoreAtPoint(f,p,state,points[i],unit(points[i]-state.point),width,0);
        total+=state.loss;}
    return total;
}
#endif
__device__ void refineDirect(Fields f,const Params& p,Seed seed,Node* nodes,int committed,
    double& widthSum,double& widthMean,int& widthCount,RefineScratch& w,RefinementStats& output) {
    if(threadIdx.x==0){
        ++output.refinementCommits;w.n=nodes[committed].depth+1;w.first=p.refineWindow?max(1,w.n-p.refineWindow):1;
        for(int at=committed;at>=0;at=nodes[at].previous){int i=nodes[at].depth;w.ids[i]=at;w.points[i]=nodes[at].point;}
    }__syncthreads();
    // Fixed reference for this solve. Update the global width accumulator only
    // after acceptance. No rescoring of immutable-prefix nodes under a new mean.
    const double referenceWidth=widthMean;
#ifndef VC_CHUNK_GPU_LIBRARY
    float auditInitial=0,auditFinal=0;bool auditHasInitial=false,auditImproved=false;
#endif
    for(int iteration=0;iteration<p.refineIterations;++iteration){
        if(threadIdx.x==0)directHistories(p,seed,nodes,w,w.points);
        __syncthreads();
        for(int i=w.first+threadIdx.x;i<w.n;i+=blockDim.x){
            float mod=0;bool valid=false;
            auto term=directTerm<true>(f,p,seed,w.points,i,w.histories[i-1],referenceWidth,w.oldWidths[i],mod,valid);
            w.valid[i]=valid&&std::isfinite(term.value);w.losses[i]=term.value;
            for(int k=0;k<3;++k)w.adjacent[k][i]={term.gradient[k*3],term.gradient[k*3+1],term.gradient[k*3+2]};
            w.historyGradient[i]={term.gradient[9],term.gradient[10],term.gradient[11]};
        }__syncthreads();
        for(int i=w.first+threadIdx.x;i<w.n;i+=blockDim.x){
            V g=w.adjacent[0][i];if(i+1<w.n)g=g+w.adjacent[1][i+1];if(i+2<w.n)g=g+w.adjacent[2][i+2];w.gradient[i]=g;
        }__syncthreads();
        if(threadIdx.x==0){
            directAdjoint(p,nodes,w);
#ifndef VC_CHUNK_GPU_LIBRARY
            if(iteration==0){++output.audit.calls;auditInitial=auditFinal=w.before;
                auditHasInitial=std::isfinite(w.before);if(!auditHasInitial)++output.audit.invalidInitial;}
            if(!w.proposal)++output.audit.noProposal;
            bool finiteGradient=true,validCurrent=true;
            for(int i=w.first;i<w.n;++i){validCurrent&=w.valid[i]!=0;
                finiteGradient&=std::isfinite(w.gradient[i].x)&&std::isfinite(w.gradient[i].y)&&std::isfinite(w.gradient[i].z);}
            output.audit.zeroGradient+=finiteGradient&&w.largest==0;
            output.audit.nonfiniteGradient+=!finiteGradient;
            output.audit.invalidCurrent+=!validCurrent;
#endif
        }__syncthreads();
        if(!w.proposal)break;
        for(int backtrack=0;backtrack<8;++backtrack){
            const float step=p.refineStep/w.largest*std::ldexp(1.f,-backtrack);
            for(int i=threadIdx.x;i<w.n;i+=blockDim.x){w.trial[i]=w.points[i];if(i<w.first)continue;
                auto& n=nodes[w.ids[i]];if(outside(n.point,p))continue;
                V d=w.points[i]-w.gradient[i]*step-n.planeOrigin;d=d-n.planeNormal*d.dot(n.planeNormal);
                float length=std::sqrt(d.dot(d));if(length>p.refineCap)d*=p.refineCap/length;
                w.trial[i]=n.planeOrigin+d;
            }__syncthreads();
            if(threadIdx.x==0)directHistories(p,seed,nodes,w,w.trial);
            __syncthreads();
            for(int i=w.first+threadIdx.x;i<w.n;i+=blockDim.x){float mod;bool valid;
                w.losses[i]=directTerm<false>(f,p,seed,w.trial,i,w.histories[i-1],referenceWidth,w.widths[i],mod,valid);
                w.valid[i]=valid&&std::isfinite(w.losses[i])&&(!outside(w.trial[i],p)||outside(w.points[i],p));
            }__syncthreads();
            if(threadIdx.x==0){float after=0;bool valid=true,moved=false;for(int i=w.first;i<w.n;++i){after+=w.losses[i];valid&=w.valid[i]!=0;moved|=(w.trial[i]-w.points[i]).dot(w.trial[i]-w.points[i])>0;}
#ifndef VC_CHUNK_GPU_LIBRARY
                ++output.audit.attempts;
                output.audit.noMovement+=!moved;
                bool clipped=false;
                for(int i=w.first;i<w.n;++i){auto& node=nodes[w.ids[i]];if(outside(node.point,p))continue;
                    V d=w.points[i]-w.gradient[i]*step-node.planeOrigin;d=d-node.planeNormal*d.dot(node.planeNormal);
                    clipped|=d.dot(d)>p.refineCap*p.refineCap;}
                output.audit.capTrials+=clipped;
                if(!valid)++output.audit.invalidTrials;
                else if(!(moved&&after<w.before))++output.audit.nonDecreasingTrials;
#endif
                if(valid&&moved&&after<w.before){
#ifndef VC_CHUNK_GPU_LIBRARY
                    auto& a=output.audit;++a.backtracks[backtrack];a.acceptedBefore+=w.before;a.acceptedAfter+=after;
                    a.minAcceptedGain=fminf(a.minAcceptedGain,w.before-after);a.maxAcceptedGain=fmaxf(a.maxAcceptedGain,w.before-after);
                    float beforeBeam=auditBeamObjective(f,p,seed,nodes,w,w.points,referenceWidth);
                    float afterBeam=auditBeamObjective(f,p,seed,nodes,w,w.trial,referenceWidth);
                    ++a.beamChecks;a.beamBefore+=beforeBeam;a.beamAfter+=afterBeam;
                    a.beamWorsened+=afterBeam>beforeBeam;
                    a.maxBeamIncrease=fmaxf(a.maxBeamIncrease,afterBeam-beforeBeam);
                    a.maxObjectiveMismatch=fmaxf(a.maxObjectiveMismatch,fmaxf(std::abs(beforeBeam-w.before),std::abs(afterBeam-after)));
                    auditFinal=after;auditImproved=true;
#endif
                    for(int i=w.first;i<w.n;++i){widthSum+=double(w.widths[i])-w.oldWidths[i];w.points[i]=w.trial[i];auto& n=nodes[w.ids[i]];
                        n.point=w.points[i];n.history=w.histories[i];V d=n.point-n.planeOrigin;
                        output.maxPlaneError=fmaxf(output.maxPlaneError,std::abs(d.dot(n.planeNormal)));
                        output.maxMovement=fmaxf(output.maxMovement,std::sqrt(d.dot(d)));
                    }
                    widthMean=widthSum/widthCount;w.accepted=1;++output.refinementAccepted;output.refinementGain+=w.before-after;
                }
            }__syncthreads();
            if(w.accepted)break;
        }
        if(!w.accepted){
#ifndef VC_CHUNK_GPU_LIBRARY
            if(threadIdx.x==0)++output.audit.exhausted;
#endif
            break;
        }
    }
    // Refresh only changed suffix scores/history for future beam continuation.
    if(threadIdx.x==0){
#ifndef VC_CHUNK_GPU_LIBRARY
        if(auditHasInitial){output.audit.initialObjective+=auditInitial;output.audit.finalObjective+=auditFinal;}
        output.audit.callsImproved+=auditImproved;
#endif
        State state=initialState(seed);int previous=w.first-1;
        if(previous>0){state.point=w.points[previous];state.previous=unit(w.points[previous]-w.points[previous-1]);
            state.current=sample(f,p,state.point,state.previous).dir;state.history=nodes[w.ids[previous]].history;
            state.historySamples=std::min(previous,std::max(1,p.historySteps));state.loss=nodes[w.ids[previous]].prefixLoss;}
        for(int i=w.first;i<w.n;++i){auto& n=nodes[w.ids[i]];
            state=scoreAtPoint(f,p,state,n.point,unit(n.point-state.point),referenceWidth,0);
            n.modulated=state.modulated;n.scoringWidth=referenceWidth;n.history=state.history;n.prefixLoss=state.loss;
        }
    }__syncthreads();
}
