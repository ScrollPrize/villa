#include "vc_object_store/objects/FiberUvExport.hpp"
#include "vc_object_store/ParallelFor.hpp"
#include "vc_object_store/Uuid.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace vc::objects {
namespace {
size_t field(const ExtensionTable& table,const std::string& name)
{
    const auto it=std::find_if(table.columns.begin(),table.columns.end(),[&](const auto& c){return c.name==name;});
    if(it==table.columns.end())throw std::logic_error("missing UV export field "+name);
    return size_t(it-table.columns.begin());
}
ExtensionValue blob(const FiberGraphUuid& uuid){return std::vector<uint8_t>(uuid.begin(),uuid.end());}
}

InitialImportResult exportFixedFiberUvColumns(const std::filesystem::path& destination,
    const std::filesystem::path& dependency,const ExtensionSource& expectedSource,
    const std::string& runUuid,const std::optional<std::string>& inputRunUuid,const std::string& method,
    const std::map<std::string,FiberGraphSetting>& settings,const FixedFiberUvColumns& in,size_t workers)
{
    const size_t n=in.uuids.size(),samples=in.sampleOwner.size();
    if(!workers||workers>size_t(INT64_MAX)||!n||!samples)
        throw std::invalid_argument("UV export requires fibers, samples and workers");
    if(in.family.size()!=n||in.winding.size()!=n||in.direction.size()!=n||
       in.sampleDs.size()!=samples||in.q.size()!=samples||in.x.size()!=samples||in.v.size()!=samples)
        throw std::invalid_argument("UV export column sizes differ");
    OpenOptions reading;reading.readOnly=true;
    auto source=ObjectCollection::open(dependency,reading);auto sourceTxn=source.read();
    const auto status=source.provisioning(sourceTxn);
    const ExtensionSource consumed{source.uuid(sourceTxn),status.generation,status.commitSequence};
    if(consumed!=expectedSource)throw std::invalid_argument("UV export input snapshot changed");
    constexpr size_t block=4096;
    const size_t blocks=(n+block-1)/block;
    auto sampleOffset=std::make_unique_for_overwrite<size_t[]>(n+1);
    sampleOffset[n]=samples;
    vc::objects::parallelFor((samples+block-1)/block,workers,[&](size_t task){
        for(size_t k=task*block;k<std::min(samples,(task+1)*block);++k){
            const auto owner=in.sampleOwner[k];
            if(owner<0||uint64_t(owner)>=n||!std::isfinite(in.sampleDs[k])||!std::isfinite(in.q[k])||
               !std::isfinite(in.x[k])||!std::isfinite(in.v[k]))
                throw std::invalid_argument("invalid UV export sample");
            const auto previous=k?in.sampleOwner[k-1]:-1;
            if(owner<previous||owner>previous+1)throw std::invalid_argument("UV samples must be grouped with every fiber present");
            if(owner!=previous)sampleOffset[owner]=k;
        }
    });
    if(in.sampleOwner.back()!=int64_t(n-1))throw std::invalid_argument("UV samples omit a fiber");
    // The only import routing data is transient: one row order and a small
    // per-block histogram. Persisted routing uses the existing UUID hash rule.
    std::vector<size_t> counts(blocks*workers,0),starts(workers+1,0);
    auto routes=std::make_unique_for_overwrite<size_t[]>(n);
    vc::objects::parallelFor(blocks,workers,[&](size_t task){
        auto* histogram=counts.data()+task*workers;
        for(size_t i=task*block;i<std::min(n,(task+1)*block);++i){
            if((in.family[i]!=0&&in.family[i]!=1)||!std::isfinite(in.winding[i])||
               std::abs(in.winding[i])>=double(INT64_MAX)||in.winding[i]!=std::round(in.winding[i])||
               (in.direction[i]!=-1&&in.direction[i]!=1))
                throw std::invalid_argument("invalid UV export fiber");
            const auto route=util::uuidHash64(in.uuids[i])%workers;routes[i]=route;++histogram[route];
        }
    });
    for(size_t shard=0;shard<workers;++shard){
        size_t count=0;for(size_t task=0;task<blocks;++task)count+=counts[task*workers+shard];
        starts[shard+1]=starts[shard]+count;
        size_t offset=starts[shard];
        for(size_t task=0;task<blocks;++task){const auto at=task*workers+shard;const auto next=counts[at];counts[at]=offset;offset+=next;}
    }
    auto order=std::make_unique_for_overwrite<size_t[]>(n);
    vc::objects::parallelFor(blocks,workers,[&](size_t task){
        auto* cursors=counts.data()+task*workers;
        for(size_t i=task*block;i<std::min(n,(task+1)*block);++i)order[cursors[routes[i]]++]=i;
    });
    InitialImportOptions options;options.shards.resize(workers);options.dependencies={dependency};
    options.store.author="fiber-workflow";options.store.edgeAdjacencyIndex=false;
    options.packed=PackedStoreOptions{};
    options.packed->residentBytes=256*1024*1024;options.packed->exportWorkers=std::min(workers,size_t(8));
    const auto stride=INT64_MAX/static_cast<int64_t>(workers);
    for(size_t i=0;i<workers;++i)options.shards[i].objectIds={1+int64_t(i)*stride,int64_t(i+1)*stride};
    for(const auto kind:{FiberGraphTable::Runs,FiberGraphTable::Inputs,FiberGraphTable::FiberResults,
                         FiberGraphTable::Widths,FiberGraphTable::Positions}){
        auto table=fiberGraphSolveTable(kind);table.basedOn={consumed};options.extensions.push_back(std::move(table));
    }
    const auto& runs=options.extensions[0];const auto& inputs=options.extensions[1];
    const auto& fibers=options.extensions[2];const auto& widths=options.extensions[3];const auto& positions=options.extensions[4];
    const auto run=fiberGraphRunRow(runUuid,"fixed_winding_uv",method,settings).values;
    const auto input=fiberGraphInputRow(runUuid,"input_0",consumed,inputRunUuid).values;
    const auto runKey=extensionUuid(runUuid);
    const auto fUuid=field(fibers,"fiber_uuid"),fRun=field(fibers,"run_uuid"),fFamily=field(fibers,"family"),
               fEnabled=field(fibers,"enabled"),fWinding=field(fibers,"selected_winding");
    const auto pUuid=field(positions,"fiber_uuid"),pRun=field(positions,"run_uuid"),pLayer=field(positions,"layer"),
               pHypothesis=field(positions,"hypothesis"),pIndex=field(positions,"node_index"),pArc=field(positions,"arc"),
               pU=field(positions,"u_turns"),pX=field(positions,"u_vx"),pV=field(positions,"v_vx");
    std::vector<ExtensionRow> widthRows; widthRows.reserve(in.widths.size());
    for(size_t i=0;i<in.widths.size();++i)widthRows.push_back(fiberGraphSolveRow(FiberGraphTable::Widths,
        {{"run_uuid",runKey},{"winding",int64_t(i)},{"width_vx",in.widths[i]}}).values);
    return importInitialCollection(destination,options,[&](size_t shard,InitialShardImport& writer){
        if(!shard){writer.appendExtensionRows(runs.name,std::span(&run,1));writer.appendExtensionRows(inputs.name,std::span(&input,1));
            writer.appendExtensionRows(widths.name,widthRows);}
        std::vector<ExtensionRow> fiberRows,positionRows;fiberRows.reserve(block);positionRows.reserve(block);
        const auto flush=[&]{
            if(!fiberRows.empty()){writer.appendExtensionRows(fibers.name,fiberRows);fiberRows.clear();}
            if(!positionRows.empty()){writer.appendExtensionRows(positions.name,positionRows);positionRows.clear();}
        };
        for(size_t at=starts[shard];at<starts[shard+1];++at){
            const auto i=order[at];const auto uuid=blob(in.uuids[i]);
            ExtensionRow fiber(fibers.columns.size());fiber[fUuid]=uuid;fiber[fRun]=runKey;
            fiber[fFamily]=in.family[i];fiber[fEnabled]=int64_t(1);fiber[fWinding]=int64_t(in.winding[i]);
            fiberRows.push_back(std::move(fiber));
            for(size_t k=sampleOffset[i];k<sampleOffset[i+1];++k){
                const auto arc=in.sampleDs[k]*in.direction[i];
                if(arc<0)throw std::invalid_argument("UV export arc must be nonnegative");
                ExtensionRow position(positions.columns.size());position[pUuid]=uuid;position[pRun]=runKey;
                position[pLayer]=int64_t(FiberGraphLayer::Refined);position[pHypothesis]=int64_t(in.winding[i]);
                position[pIndex]=int64_t(k-sampleOffset[i]);position[pArc]=arc;
                position[pU]=in.q[k]-in.winding[i];position[pX]=in.x[k];position[pV]=in.v[k];
                positionRows.push_back(std::move(position));if(positionRows.size()>=block)flush();
            }
            if(fiberRows.size()>=block)flush();
        }
        flush();
    });
}
} // namespace vc::objects
