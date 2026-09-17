#include "GraphInternal.hpp"
#include <fstream>
#include <iomanip>
#include <sstream>
#include <set>
#include <bit>

namespace vc::constraints {
namespace {
using Json=nlohmann::json;
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
nlohmann::json graphToJson(const Graph& graph) {
    Json out={{"type","vc3d_constraint_graph"},{"version",1},{"settings",settings(graph.settings)},{"seed",graph.seed},
        {"nodes",Json::array()},{"links",Json::array()}};
    for(const auto& [id,p]:graph.nodes) {
        const auto& n=*p.source;Json node={{"id",id},{"path",n.path.string()},{"fiber",n.fiber},{"direction",std::string(1,n.direction)},
            {"rows",n.rows},{"cols",n.cols},{"xyz",Json::array()},{"uv",Json::array()},{"initial_uv",Json::array()},
            {"line",Json::array()},{"controls",n.controls},{"line_arclength",n.lineArclength},{"column_arclength",n.columnArclength}};
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
FitResult loadSession(const std::filesystem::path& path) {
    std::ifstream input(path);if(!input)throw std::runtime_error("Cannot open graph session");
    auto json=Json::parse(input);auto graph=graphFromJson(json);
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
} // namespace vc::constraints
