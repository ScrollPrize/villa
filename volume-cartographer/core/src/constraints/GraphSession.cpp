#include "GraphInternal.hpp"
#include "Quarantine.hpp"
#include <fstream>
#include <iomanip>
#include <sstream>
#include <set>
#include <bit>
#include <cstring>
#include <chrono>

namespace vc::constraints {
namespace {
using Json=nlohmann::json;
std::string revision(const std::string& text) {
    uint64_t hash=14695981039346656037ULL;
    for(unsigned char c:text){hash^=c;hash*=1099511628211ULL;}
    std::ostringstream out;out<<std::hex<<hash;return out.str();
}
std::string readText(const std::filesystem::path& path) {
    std::ifstream input(path,std::ios::binary);
    if(!input)throw std::runtime_error("Cannot open graph session: "+path.string());
    std::ostringstream text;text<<input.rdbuf();
    if(input.bad())throw std::runtime_error("Cannot read graph session");
    return text.str();
}
void writeText(const std::filesystem::path& path,const std::string& text) {
    std::ofstream output(path,std::ios::binary|std::ios::trunc);output<<text;output.flush();
    if(!output)throw std::runtime_error("Cannot write graph session: "+path.string());
}
template<int N> Json vec(const cv::Vec<double,N>& v) {Json a=Json::array();for(int i=0;i<N;++i)a.push_back(v[i]);return a;}
template<int N> cv::Vec<double,N> vector(const Json& j) {
    if(!j.is_array()||j.size()!=N)throw std::runtime_error("Invalid coordinate array in graph session");
    cv::Vec<double,N> v;for(int i=0;i<N;++i){v[i]=j[i].get<double>();if(!std::isfinite(v[i]))throw std::runtime_error("Nonfinite session coordinate");}return v;
}
Json location(const Location& p){return {{"node",p.node},{"grid",vec(p.grid)},{"line_index",p.lineIndex}};}
Location location(const Json& j){return {j.at("node").get<std::string>(),vector<2>(j.at("grid")),j.value("line_index",int64_t{-1})};}
Json settings(const Settings& s) {
    return {{"spacing",s.spacing},{"radius",s.radius},{"fiber_scale",s.fiberScale},{"patch_scale",s.patchScale},
        {"normal_scale",s.normalScale},{"volume_scale",s.volumeScale},{"stiffness",s.stiffness},{"normal_manifest",s.normalManifest},
        {"fiber_folder",s.fiberFolder.string()},{"patch_folder",s.patchFolder.string()}};
}
Settings settings(const Json& j) {
    Settings s;s.spacing=j.at("spacing");s.radius=j.at("radius");s.fiberScale=j.at("fiber_scale");s.patchScale=j.at("patch_scale");
    s.normalScale=j.at("normal_scale");s.volumeScale=j.at("volume_scale");s.stiffness=j.at("stiffness");
    s.normalManifest=j.at("normal_manifest");s.fiberFolder=j.at("fiber_folder").get<std::string>();s.patchFolder=j.at("patch_folder").get<std::string>();s.validate();return s;
}
std::string surfaceChecksum(const Json& document,const cv::Mat_<cv::Vec3f>& grid,cv::Vec2d origin) {
    uint64_t hash=14695981039346656037ULL;
    auto byte=[&](unsigned char c){hash^=c;hash*=1099511628211ULL;};
    for(unsigned char c:document.dump())byte(c);
    for(unsigned char c:vec(origin).dump())byte(c);
    auto word=[&](uint32_t value){for(int shift=0;shift<32;shift+=8)byte(static_cast<unsigned char>(value>>shift));};
    word(uint32_t(grid.rows));word(uint32_t(grid.cols));
    for(const auto& point:grid)for(float value:point.val)word(std::bit_cast<uint32_t>(value));
    std::ostringstream out;out<<std::hex<<hash;return out.str();
}

}
nlohmann::json settingsToJson(const Settings& s) { return settings(s); }
Settings settingsFromJson(const nlohmann::json& j) { return settings(j); }
nlohmann::json locationToJson(const Location& l) { return location(l); }
Location locationFromJson(const nlohmann::json& j) { return location(j); }
nlohmann::json graphToJson(const Graph& graph) {
    Json out={{"type","vc3d_constraint_graph"},{"version",1},{"settings",settings(graph.settings)},{"seed",graph.seed},
        {"nodes",Json::array()},{"links",Json::array()}};
    for(const auto& [id,p]:graph.nodes) {
        const auto& n=*p.source;Json node={{"id",id},{"path",n.path.string()},{"fiber",n.fiber},{"direction",std::string(1,n.direction)},
            {"rows",n.rows},{"cols",n.cols},{"xyz",Json::array()},{"uv",Json::array()},{"initial_uv",Json::array()},
            {"line",Json::array()},{"controls",n.controls},{"line_arclength",n.lineArclength},{"column_arclength",n.columnArclength}};
        if(n.fiber)node["strip_generation"]=n.stripGeneration;
        for(const auto& xyz:n.xyz)node["xyz"].push_back(vec(xyz));
        for(const auto& uv:p.uv)node["uv"].push_back(vec(uv));
        for(const auto& uv:n.initialUv)node["initial_uv"].push_back(vec(uv));
        for(const auto& xyz:n.line)node["line"].push_back(vec(xyz));
        out["nodes"].push_back(std::move(node));
    }
    for(const auto& l:graph.links)out["links"].push_back({{"a",location(l.a)},{"b",location(l.b)},{"keypoint",l.keypoint},{"pending",l.pending}});
    return out;
}
Graph graphFromJson(const nlohmann::json& json) {
    if(json.at("type")!="vc3d_constraint_graph"||json.at("version")!=1)throw std::runtime_error("Unsupported constraint graph session");
    Graph g;g.settings=settings(json.at("settings"));g.seed=json.at("seed");
    for(const auto& item:json.at("nodes")) {
        auto n=std::make_shared<Node>();n->id=item.at("id");n->path=item.at("path").get<std::string>();n->fiber=item.at("fiber");
        auto direction=item.at("direction").get<std::string>();if(direction.size()!=1)throw std::runtime_error("Invalid fiber direction");n->direction=direction[0];
        n->rows=item.at("rows");n->cols=item.at("cols");
        n->stripGeneration=item.value("strip_generation",0);
        if(n->rows<2||n->cols<2||int64_t(n->rows)*n->cols>16000000)throw std::runtime_error("Invalid source grid dimensions");
        for(const auto& v:item.at("xyz"))n->xyz.push_back(vector<3>(v));
        for(const auto& v:item.at("initial_uv"))n->initialUv.push_back(vector<2>(v));
        Placement p;p.source=n;for(const auto& v:item.at("uv"))p.uv.push_back(vector<2>(v));
        if(n->xyz.size()!=size_t(n->rows)*n->cols||p.uv.size()!=n->xyz.size()||n->initialUv.size()!=n->xyz.size())throw std::runtime_error("Source grid size mismatch");
        for(const auto& v:item.at("line"))n->line.push_back(vector<3>(v));
        n->controls=item.at("controls").get<std::vector<size_t>>();
        n->lineArclength=item.at("line_arclength").get<std::vector<double>>();n->columnArclength=item.at("column_arclength").get<std::vector<double>>();
        if(n->fiber) {
            if(n->lineArclength.size()!=n->line.size()||n->columnArclength.size()!=size_t(n->cols)||n->controls.size()<2)
                throw std::runtime_error("Invalid fiber source mapping");
            for(auto control:n->controls)if(control>=n->line.size())throw std::runtime_error("Invalid fiber control index");
            for(const auto* values:{&n->lineArclength,&n->columnArclength}) {
                if(!std::is_sorted(values->begin(),values->end()))throw std::runtime_error("Unordered source mapping");
                for(double v:*values)if(!std::isfinite(v)||v<0)throw std::runtime_error("Invalid source arclength");
            }
        }
        detail::triangulate(*n);if(n->triangles.empty())throw std::runtime_error("Source has no valid quads");
        if(!g.nodes.emplace(n->id,std::move(p)).second)throw std::runtime_error("Duplicate graph node");
    }
    if(!g.nodes.contains(g.seed))throw std::runtime_error("Session seed is missing");
    for(const auto& item:json.at("links")) {
        Link l{location(item.at("a")),location(item.at("b")),item.at("keypoint").get<bool>(),item.value("pending",false)};
        (void)placedLocation(g,l.a);(void)placedLocation(g,l.b);g.links.push_back(std::move(l));
    }
    std::set<std::string> connected{g.seed};bool changed=true;
    while(changed){changed=false;for(const auto& l:g.links){if(connected.contains(l.a.node))changed|=connected.insert(l.b.node).second;if(connected.contains(l.b.node))changed|=connected.insert(l.a.node).second;}}
    if(connected.size()!=g.nodes.size())throw std::runtime_error("Saved graph is disconnected");
    return g;
}
void exportSurface(const FitResult& result,const std::filesystem::path& path) {
    if(!result.valid||!result.surface)throw std::runtime_error("Only a valid committed surface can be exported");
    // save() mutates a QuadSurface's backing path: export a private copy so
    // viewers and concurrent snapshots never observe those mutations.
    auto copy=std::make_shared<QuadSurface>(*result.surface->rawPointsPtr(),result.surface->scale());
    copy->setStrictQuadRenderValidity(true);copy->save(path,path.filename().string(),true);
    copy->writeValidMask();
    auto metaPath=path/"meta.json";std::ifstream input(metaPath);auto meta=Json::parse(input);input.close();
    meta["constraint_graph"]={{"origin",vec(result.origin)},{"coordinate_frame","patch"},
        {"fiber_scale",result.graph.settings.fiberScale},{"patch_scale",result.graph.settings.patchScale},
        {"spacing",result.graph.settings.spacing},{"includes_fiber_ribbons",true}};
    std::ofstream output(metaPath,std::ios::trunc);output<<meta.dump(2);output.flush();if(!output)throw std::runtime_error("Failed to write surface metadata");
}
void saveSession(const FitResult& result,const std::filesystem::path& path) {
    if(!result.graph.windingLayout.is_null())throw std::runtime_error("Winding layouts are read-only; reopen the generated workspace file or export tifxyz");
    if(!result.valid)throw std::runtime_error("Cannot save an invalid graph");
    if(!result.surface)throw std::runtime_error("Cannot save a missing surface");
    auto document=graphToJson(result.graph);
    const auto checksum=surfaceChecksum(document,*result.surface->rawPointsPtr(),result.origin);
    auto artifact=path.stem().string()+"."+checksum+".tifxyz";
    std::filesystem::create_directories(std::filesystem::absolute(path).parent_path());
    exportSurface(result,path.parent_path()/artifact);
    document["surface"]=artifact;document["origin"]=vec(result.origin);document["surface_checksum"]=checksum;
    auto temporary=path;temporary+=".tmp";
    try {
        std::ofstream output(temporary,std::ios::trunc);output<<document.dump();output.flush();
        if(!output)throw std::runtime_error("Failed to write graph session");output.close();
        std::filesystem::rename(temporary,path);
    }catch(...){std::error_code ec;std::filesystem::remove(temporary,ec);throw;}
}

namespace {
FitResult loadWindingLayout(const Json& json,const std::filesystem::path& path) {
    const int version=json.at("version");
    if((version!=1&&version!=2)||json.at("geometry_format")!="little_endian_float64_xyzuv")
        throw std::runtime_error("Unsupported winding layout format");
    Graph graph;graph.settings=settings(json.at("settings"));graph.seed=json.at("seed");
    graph.windingLayout=json.at("layout");
    if(!graph.windingLayout.is_object()||!graph.windingLayout.value("read_only",false))
        throw std::runtime_error("Winding layout must be read-only");
    const std::filesystem::path name=json.at("geometry_file").get<std::string>();
    if(name.empty()||name.is_absolute()||name.has_parent_path()||name=="."||name=="..")
        throw std::runtime_error("Winding geometry must be a sibling file");
    const auto binary=path.parent_path()/name;
    const auto bytes=std::filesystem::file_size(binary);
    if(bytes!=json.at("geometry_bytes").get<uint64_t>())throw std::runtime_error("Winding geometry size mismatch");
    std::ifstream input(binary,std::ios::binary);
    if(!input)throw std::runtime_error("Cannot open winding geometry");
    uint64_t consumed=0;
    for(const auto& source:json.at("sources")) {
        auto node=std::make_shared<Node>();node->id=source.at("id");node->path=source.at("path").get<std::string>();
        node->fiber=source.at("fiber");node->rows=source.at("rows");node->cols=source.at("cols");
        const int64_t count=int64_t(node->rows)*node->cols;
        if(node->rows<1||node->cols<1||count>16000000||(node->fiber&&(node->rows!=3||node->cols<2)))
            throw std::runtime_error("Invalid winding source dimensions");
        const uint64_t length=uint64_t(count)*5*sizeof(double);
        if(source.at("byte_offset").get<uint64_t>()!=consumed||consumed>bytes||length>bytes-consumed)
            throw std::runtime_error("Invalid winding geometry range");
        if(source.value("quarantined",false)) {
            if(node->fiber)throw std::runtime_error("Fibers cannot be quarantined as patches");
            input.seekg(std::streamoff(length),std::ios::cur);
            if(!input)throw std::runtime_error("Truncated quarantined geometry");
            consumed+=length;continue;
        }
        std::vector<double> values(size_t(count)*5);
        input.read(reinterpret_cast<char*>(values.data()),std::streamsize(length));
        if(!input)throw std::runtime_error("Truncated winding geometry");
        consumed+=length;
        for(double& value:values) {
            if constexpr(std::endian::native!=std::endian::little) {
                auto* begin=reinterpret_cast<unsigned char*>(&value);std::reverse(begin,begin+sizeof(double));
            }
            if(!std::isfinite(value))throw std::runtime_error("Nonfinite winding geometry");
        }
        node->xyz.reserve(count);node->initialUv.reserve(count);
        for(int64_t i=0;i<count;++i) {
            const double* v=values.data()+i*5;node->xyz.push_back({v[0],v[1],v[2]});node->initialUv.push_back({v[3],v[4]});
        }
        if(node->fiber) {
            node->controls={0,size_t(node->cols-1)};
            for(int i=0;i<node->cols;++i) {
                node->line.push_back(node->xyz[node->cols+i]);
                const double arc=i?node->lineArclength.back()+cv::norm(node->line[i]-node->line[i-1]):0;
                node->lineArclength.push_back(arc);node->columnArclength.push_back(arc);
            }
        }
        if(version==2&&!node->fiber) {
            const size_t cells=size_t(node->rows-1)*(node->cols-1);
            node->quadMask.assign(cells,0);size_t end=0;
            const auto& runs=source.at("quad_runs");
            if(!runs.is_array())throw std::runtime_error("Invalid winding quad runs");
            for(const auto& run:runs) {
                if(!run.is_array()||run.size()!=2||!run[0].is_number_integer()||!run[1].is_number_integer())
                    throw std::runtime_error("Invalid winding quad run");
                const int64_t start=run[0],length=run[1];
                if(start<0||length<=0||uint64_t(start)<end||uint64_t(start)>cells||uint64_t(length)>cells-uint64_t(start))
                    throw std::runtime_error("Invalid winding quad range");
                end=size_t(start+length);
                std::fill(node->quadMask.begin()+start,node->quadMask.begin()+end,1);
            }
            for(int y=0;y+1<node->rows;++y)for(int x=0;x+1<node->cols;++x)if(node->quadMask[y*(node->cols-1)+x]) {
                const int a=y*node->cols+x;
                for(int i:{a,a+1,a+node->cols,a+node->cols+1})if(!detail::valid(node->xyz[i]))
                    throw std::runtime_error("Winding quad has a missing vertex");
            }
        }
        detail::triangulate(*node);
        if(!graph.nodes.emplace(node->id,Placement{node,node->initialUv}).second)
            throw std::runtime_error("Duplicate winding source");
    }
    if(consumed!=bytes||!graph.nodes.contains(graph.seed))throw std::runtime_error("Incomplete winding layout");
    return renderPlacement(std::move(graph));
}
}

FitResult loadSession(const std::filesystem::path& path) {
    const auto text=readText(path);auto json=Json::parse(text);
    if(json.value("type",std::string{})=="vc3d_winding_layout") {
        auto result=loadWindingLayout(json,path);result.sessionRevision=revision(text);return result;
    }
    auto graph=graphFromJson(json);
    // Local solves deliberately retain distant XYZ samples. Restore the exact
    // accepted grid instead of globally relaxing it again during reopening.
    if(json.contains("surface_checksum")&&json.contains("surface")&&json.contains("origin")) {
        try {
            auto surface=std::make_shared<QuadSurface>(path.parent_path()/json.at("surface").get<std::string>());
            const auto origin=vector<2>(json.at("origin"));
            const auto& grid=*surface->rawPointsPtr();
            if(surfaceChecksum(graphToJson(graph),grid,origin)==json.at("surface_checksum").get<std::string>()) {
                auto restored=detail::restoreSurface(graph,grid,origin);
                if(restored.valid)return restored;
            }
        }catch(const std::exception&) { /* Missing/damaged derived data can be rebuilt. */ }
    }
    return render(std::move(graph));
}
std::filesystem::path quarantineDirectory(const std::filesystem::path& workspace) {
    const auto parent=std::filesystem::absolute(workspace).parent_path();
    return parent.parent_path()/(parent.filename().string()+"_quarantined");
}

namespace detail {
std::string workspaceText(const std::filesystem::path& path){return readText(path);}
std::string workspaceRevision(const std::string& text){return revision(text);}
QuarantineMoves planQuarantine(const std::filesystem::path& root,const std::filesystem::path& workspace,
                              const std::vector<std::filesystem::path>& sources) {
    namespace fs=std::filesystem;
    const auto patchRoot=fs::canonical(root),destination=quarantineDirectory(workspace);
    QuarantineMoves moves;
    for(const auto& folder:sources) {
        const auto source=fs::canonical(folder);
        const auto relative=source.lexically_relative(patchRoot);
        if(relative.empty()||relative=="."||*relative.begin()==".."||!fs::is_directory(source))
            throw std::runtime_error("Selected patch is not a source folder inside the patch catalog");
        const auto target=destination/source.filename();
        if(fs::exists(target))throw std::runtime_error("Quarantine destination already exists: "+target.string());
        for(const auto& [from,to]:moves)if(to==target&&from!=source)
            throw std::runtime_error("Selected patches have the same quarantine folder name");
        moves.emplace(source,target);
    }
    return moves;
}
void publishQuarantine(const std::filesystem::path& path,const std::string& original,
                       const std::string& candidate,const QuarantineMoves& moves) {
    namespace fs=std::filesystem;
    const auto destination=quarantineDirectory(path);
    // A per-workspace lock and journal protect concurrent app instances. A
    // failed move/publication restores moved folders and leaves the manifest.
    const auto lock=fs::path(path.string()+".quarantine-lock");
    if(!fs::create_directory(lock))throw std::runtime_error("Another quarantine transaction is active: "+lock.string());
    std::vector<std::pair<fs::path,fs::path>> moved;
    bool published=false;
    try {
        if(readText(path)!=original)throw std::runtime_error("The workspace changed while preparing quarantine; reopen it");
        const auto temporary=lock/"workspace.json";
        writeText(temporary,candidate);writeText(lock/"original.json",original);
        Json journal={{"workspace",path.string()},{"moves",Json::array()}};
        for(const auto& [from,to]:moves)journal["moves"].push_back({{"from",from.string()},{"to",to.string()}});
        writeText(lock/"moves.json",journal.dump(2)+"\n");
        const auto stamp=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        auto backup=path.parent_path()/(path.stem().string()+".before-quarantine-"+std::to_string(stamp)+".json");
        fs::copy_file(lock/"original.json",backup,fs::copy_options::none);
        fs::create_directories(destination);
        for(const auto& [from,to]:moves) {
            if(fs::exists(to))throw std::runtime_error("Quarantine destination appeared during preparation: "+to.string());
            fs::rename(from,to);moved.emplace_back(from,to);
        }
        if(readText(path)!=original)throw std::runtime_error("The workspace changed during quarantine; restoring source folders");
        fs::rename(temporary,path);published=true;
    }catch(...) {
        const auto failure=std::current_exception();
        bool rollbackFailed=false;
        for(auto it=moved.rbegin();it!=moved.rend();++it) {
            std::error_code error;fs::rename(it->second,it->first,error);rollbackFailed|=bool(error);
        }
        if(rollbackFailed)throw std::runtime_error("Quarantine rollback needs recovery; original manifest and move journal retained at "+lock.string());
        std::error_code ignored;fs::remove_all(lock,ignored);std::rethrow_exception(failure);
    }
    if(published){std::error_code ignored;fs::remove_all(lock,ignored);}
}
} // namespace detail

FitResult quarantinePatches(const FitResult& current,const std::filesystem::path& workspace,
                            const std::vector<std::string>& sourceIds,const Progress& progress) {
    namespace fs=std::filesystem;
    if(!current.valid||current.graph.windingLayout.is_null()||sourceIds.empty())
        throw std::runtime_error("Select a patch in a saved winding workspace to quarantine");
    const auto path=fs::absolute(workspace),destination=quarantineDirectory(path);
    const auto original=readText(path);auto document=Json::parse(original);
    if(current.sessionRevision.empty()||current.sessionRevision!=revision(original))
        throw std::runtime_error("The workspace changed on disk; reopen it before quarantining a patch");
    if(document.at("type")!="vc3d_winding_layout")throw std::runtime_error("Not a winding workspace");
    std::vector<fs::path> sources;
    for(const auto& id:sourceIds) {
        auto selected=current.graph.nodes.find(id);
        if(selected==current.graph.nodes.end()||selected->second.source->fiber)
            throw std::runtime_error("Quarantine requires a selected patch");
        sources.push_back(selected->second.source->path);
    }
    const auto moves=detail::planQuarantine(current.graph.settings.patchFolder,path,sources);
    // The manifest is self-contained: retain payload offsets and original
    // paths for audit/recovery, but exclude every component of each moved file.
    size_t excluded=0;std::string nextSeed;
    for(auto& source:document.at("sources")) {
        if(!source.value("quarantined",false)&&!source.at("fiber").get<bool>()) {
            const auto from=fs::weakly_canonical(source.at("path").get<std::string>());
            if(auto found=moves.find(from);found!=moves.end()) {
                source["quarantined"]=true;source["quarantine_path"]=found->second.string();++excluded;
            }
        }
        if(!source.value("quarantined",false)) {
            const auto id=source.at("id").get<std::string>();
            if(nextSeed.empty()||id==document.at("seed").get<std::string>())nextSeed=id;
        }
    }
    if(excluded< moves.size()||nextSeed.empty())throw std::runtime_error("Quarantine would leave no usable workspace");
    document["seed"]=nextSeed;
    const auto candidate=document.dump(2)+"\n";
    if(progress)progress("Rebuilding the surface without "+std::to_string(excluded)+" quarantined patch components");
    auto graph=current.graph;graph.seed=nextSeed;
    for(const auto& source:document.at("sources"))if(source.value("quarantined",false))
        graph.nodes.erase(source.at("id").get<std::string>());
    auto result=renderPlacement(std::move(graph),nullptr,progress);
    if(!result.valid)throw std::runtime_error("Quarantine would leave an invalid surface: "+result.message);
    result.sessionRevision=revision(candidate);
    if(progress)progress("Saving the workspace and moving source folders to "+destination.string());
    detail::publishQuarantine(path,original,candidate,moves);
    result.message="Quarantined "+std::to_string(moves.size())+" patch folder(s) in "+destination.string()+"; "+result.message;
    return result;
}

} // namespace vc::constraints
