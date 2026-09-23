#include "GraphInternal.hpp"
#include "vc/constraints/FiberReview.hpp"
#include "vc/lasagna/LineModel.hpp"
#include "Quarantine.hpp"
#include <fstream>

namespace vc::constraints {
namespace {
double columnAt(const std::vector<double>& arc,double position) {
    auto it=std::lower_bound(arc.begin(),arc.end(),position);
    if(it==arc.begin())return 0.;
    if(it==arc.end())return double(arc.size()-1);
    const size_t hi=size_t(it-arc.begin());
    return hi-1+(position-arc[hi-1])/std::max(arc[hi]-arc[hi-1],1e-12);
}
double arcAt(const std::vector<double>& arc,double column) {
    column=std::clamp(column,0.,double(arc.size()-1));
    const size_t lo=std::min(size_t(column),arc.size()-2);const double f=column-lo;
    return arc[lo]*(1-f)+arc[lo+1]*f;
}
}
void detail::remapFiberLocation(Location& location,const Node& oldStrip,const Node& newStrip) {
    if(location.node!=oldStrip.id)return;
    if(oldStrip.rows==newStrip.rows&&oldStrip.cols==newStrip.cols&&oldStrip.columnArclength==newStrip.columnArclength)return;
    if(location.lineIndex>=0){location.grid=fiberLineLocation(newStrip,size_t(location.lineIndex)).grid;return;}
    location.grid={columnAt(newStrip.columnArclength,arcAt(oldStrip.columnArclength,location.grid[0])),
                   location.grid[1]*(newStrip.rows-1)/double(oldStrip.rows-1)};
}
void detail::replaceFiberStrip(Graph& graph,const std::shared_ptr<const Node>& replacement) {
    auto found=graph.nodes.find(replacement->id);if(found==graph.nodes.end())return;
    const auto old=found->second.source;if(old==replacement)return;
    if(old->rows==replacement->rows&&old->cols==replacement->cols&&old->columnArclength==replacement->columnArclength) {
        found->second.source=replacement;return; // preserve stored UV and links bit-for-bit
    }
    std::vector<cv::Vec2d> uv;uv.reserve(replacement->xyz.size());
    for(int row=0;row<replacement->rows;++row)for(int col=0;col<replacement->cols;++col) {
        const cv::Vec2d at{columnAt(old->columnArclength,replacement->columnArclength[col]),row*(old->rows-1)/double(replacement->rows-1)};
        uv.push_back(interpolate(found->second.uv,weights(*old,at)));
    }
    found->second={replacement,std::move(uv)};
    for(auto& link:graph.links){remapFiberLocation(link.a,*old,*replacement);remapFiberLocation(link.b,*old,*replacement);}
}
bool regenerateFiberStrips(Graph& graph,const lasagna::NormalSampler& sampler,const std::atomic_bool* cancel) {
    Graph candidate=graph;bool changed=false;
    for(const auto& [id,p]:graph.nodes)if(p.source->fiber&&p.source->stripGeneration!=kFiberStripGeneration) {
        detail::cancelled(cancel);const auto& n=*p.source;
        Fiber fiber;fiber.id=id;fiber.path=n.path;fiber.direction=n.direction;fiber.line=n.line;fiber.controls=n.controls;
        detail::replaceFiberStrip(candidate,makeFiberNode(fiber,graph.settings,sampler,cancel));changed=true;
    }
    if(changed)graph=std::move(candidate);return changed;
}
bool regenerateReviewFiber(ReviewFiber& fiber,const lasagna::NormalSampler& sampler,const std::atomic_bool* cancel) {
    if(fiber.base.nodes.empty())return false;
    ReviewFiber candidate=fiber;const auto previous=fiber.base.nodes.at(fiber.base.seed).source;
    if(!regenerateFiberStrips(candidate.base,sampler,cancel))return false;
    const auto replacement=candidate.base.nodes.at(candidate.base.seed).source;
    if(candidate.pending)detail::replaceFiberStrip(*candidate.pending,replacement);
    for(auto& [id,decision]:candidate.decisions)detail::replaceFiberStrip(decision.alignment,replacement);
    for(auto& proposal:candidate.proposals)for(auto& link:proposal.contacts) {
        detail::remapFiberLocation(link.a,*previous,*replacement);detail::remapFiberLocation(link.b,*previous,*replacement);
    }
    fiber=std::move(candidate);return true;
}
} // namespace vc::constraints

namespace vc::constraints {
std::string fiberStripCacheKey(const Fiber& f,const Settings& s,const std::string& method) {
    std::ifstream input(f.path,std::ios::binary);
    if(!input)throw std::runtime_error("Cannot read fiber "+f.path.string());
    std::string text{std::istreambuf_iterator<char>(input),std::istreambuf_iterator<char>()};
    nlohmann::json key={{"strip_version",kFiberStripGeneration},{"cross_samples",7},{"spacing",s.spacing},{"fiber_scale",s.fiberScale},
        {"normal_scale",s.normalScale},{"normal_manifest",s.normalManifest},{"direction",std::string(1,f.direction)},{"controls",f.controls},{"line_points",f.line.size()},{"method",method}};
    return detail::workspaceRevision(key.dump()+"\n"+text);
}
namespace {
std::filesystem::path cacheEntry(const std::filesystem::path& directory,const std::string& key){return directory/(key+".strip.json");}
}
std::shared_ptr<const Node> loadCachedFiberNode(const std::filesystem::path& directory,const Fiber& f,const Settings& s,const std::string& method) {
    const auto key=fiberStripCacheKey(f,s,method);
    std::ifstream input(cacheEntry(directory,key));if(!input)return nullptr;
    nlohmann::json j;
    try{j=nlohmann::json::parse(input);}catch(const std::exception&){return nullptr;}
    if(j.value("key","")!=key||j.value("type","")!="vc_fiber_strip_cache")return nullptr;
    auto node=std::make_shared<Node>();node->id=f.id;node->path=f.path;node->fiber=true;node->direction=f.direction;
    node->line=f.line;node->controls=f.controls;node->stripGeneration=j.at("strip_generation");
    node->rows=j.at("rows");node->cols=j.at("cols");
    for(const auto& p:j.at("xyz"))node->xyz.push_back({p[0].get<double>(),p[1].get<double>(),p[2].get<double>()});
    for(const auto& p:j.at("initial_uv"))node->initialUv.push_back({p[0].get<double>(),p[1].get<double>()});
    node->lineArclength=j.at("line_arclength").get<std::vector<double>>();node->columnArclength=j.at("column_arclength").get<std::vector<double>>();
    if(int(node->xyz.size())!=node->rows*node->cols||node->initialUv.size()!=node->xyz.size()||node->lineArclength.size()!=f.line.size()||int(node->columnArclength.size())!=node->cols)return nullptr;
    detail::triangulate(*node);
    if(node->triangles.empty())return nullptr;
    return node;
}
void saveCachedFiberNode(const std::filesystem::path& directory,const Fiber& f,const Settings& s,const Node& node,const std::string& method) {
    std::filesystem::create_directories(directory);
    const auto key=fiberStripCacheKey(f,s,method);
    nlohmann::json j={{"type","vc_fiber_strip_cache"},{"version",1},{"key",key},{"fiber",f.path.filename().string()},{"strip_generation",node.stripGeneration},
        {"rows",node.rows},{"cols",node.cols},{"xyz",nlohmann::json::array()},{"initial_uv",nlohmann::json::array()},
        {"line_arclength",node.lineArclength},{"column_arclength",node.columnArclength}};
    for(const auto& p:node.xyz)j["xyz"].push_back({p[0],p[1],p[2]});
    for(const auto& p:node.initialUv)j["initial_uv"].push_back({p[0],p[1]});
    const auto target=cacheEntry(directory,key);auto temporary=target;temporary+="."+std::to_string(uint64_t(std::hash<std::string>{}(f.id)))+".tmp";
    {std::ofstream output(temporary,std::ios::trunc);output<<j;if(!output)throw std::runtime_error("Failed to write strip cache "+target.string());}
    std::filesystem::rename(temporary,target);
}
} // namespace vc::constraints

namespace vc::constraints {
namespace {
// Interpolate the entries of `values` flagged bad from the nearest good
// neighbours along the columns (linear, then renormalized when `unitize`).
// Runs at either end copy the nearest good entry.
void interpolateBad(std::vector<cv::Vec3d>& values,std::vector<char>& bad,bool unitize) {
    const int n=int(values.size());
    int firstGood=-1;for(int i=0;i<n;++i)if(!bad[i]){firstGood=i;break;}
    if(firstGood<0)return; // nothing to interpolate from; caller falls back
    for(int i=0;i<firstGood;++i)values[i]=values[firstGood];
    int lastGood=firstGood;
    for(int i=firstGood+1;i<n;++i) {
        if(bad[i])continue;
        for(int j=lastGood+1;j<i;++j) {
            const double f=double(j-lastGood)/(i-lastGood);
            values[j]=values[lastGood]*(1-f)+values[i]*f;
            if(unitize)values[j]=detail::unit(values[j]);
        }
        lastGood=i;
    }
    for(int i=lastGood+1;i<n;++i)values[i]=values[lastGood];
    std::fill(bad.begin(),bad.end(),0);
}
std::vector<cv::Vec3d> gaussianAlongColumns(const std::vector<cv::Vec3d>& values,double sigma) {
    const int n=int(values.size());std::vector<cv::Vec3d> out(n);
    const int radius=int(std::ceil(3*sigma));
    for(int i=0;i<n;++i) {
        cv::Vec3d sum{0,0,0};double weight=0;
        for(int k=-radius;k<=radius;++k){const int j=i+k;if(j<0||j>=n)continue;const double w=std::exp(-.5*k*k/(sigma*sigma));sum+=values[j]*w;weight+=w;}
        out[i]=sum/weight;
    }
    return out;
}
}
std::shared_ptr<const Node> makeFastFiberNode(const Fiber& f,const Settings& s,const lasagna::NormalSampler& sampler,const std::atomic_bool* cancel,const FastStripOptions& o) {
    if(f.line.size()<2)throw std::runtime_error("Fiber "+f.id+" has fewer than two line points");
    const double spacing=s.spacing;constexpr int rows=7;const int middle=rows/2;
    // Dense arclength and uniform columns; the final column lands on the end
    // point so the whole fiber maps into the grid.
    std::vector<double> arc(f.line.size(),0.);
    for(size_t i=1;i<f.line.size();++i)arc[i]=arc[i-1]+cv::norm(f.line[i]-f.line[i-1]);
    std::vector<double> columns;for(double a=0;a<arc.back();a+=spacing)columns.push_back(a);
    if(columns.size()<2||arc.back()-columns.back()>spacing*.25)columns.push_back(arc.back());else columns.back()=arc.back();
    const int cols=int(columns.size());
    auto at=[&](double a){auto it=std::lower_bound(arc.begin(),arc.end(),a);if(it==arc.begin())return f.line.front();if(it==arc.end())return f.line.back();
        const size_t hi=size_t(it-arc.begin());const double span=arc[hi]-arc[hi-1];const double t=span>0?(a-arc[hi-1])/span:0;return f.line[hi-1]*(1-t)+f.line[hi]*t;};
    std::vector<cv::Vec3d> center(cols),tangent(cols),normal(cols);std::vector<char> badNormal(cols,0);
    for(int c=0;c<cols;++c)center[c]=at(columns[c]);
    for(int c=0;c<cols;++c){const auto& a=center[std::max(c-1,0)];const auto& b=center[std::min(c+1,cols-1)];tangent[c]=detail::unit(b-a);}
    bool any=false;
    for(int c=0;c<cols;++c) {
        detail::cancelled(cancel);
        const auto sample=sampler.sampleNormal(center[c]);
        badNormal[c]=!sample.valid||cv::norm(sample.normal)<1e-6;
        normal[c]=badNormal[c]?cv::Vec3d{0,0,0}:detail::unit(sample.normal);any|=!badNormal[c];
    }
    if(!any)throw std::runtime_error("No Lasagna normals along fiber "+f.id);
    // Consistent sign along the fiber, then flag normals that disagree with
    // their neighbourhood (folded or noisy field) and interpolate them.
    int previous=-1;
    for(int c=0;c<cols;++c){if(badNormal[c])continue;if(previous>=0&&normal[c].dot(normal[previous])<0)normal[c]=-normal[c];previous=c;}
    interpolateBad(normal,badNormal,true);
    {
        auto smoothed=gaussianAlongColumns(normal,o.normalSigma);for(auto& n:smoothed)n=detail::unit(n);
        for(int c=0;c<cols;++c)if(normal[c].dot(smoothed[c])<std::cos(o.outlierAngle*CV_PI/180))badNormal[c]=1;
        interpolateBad(normal,badNormal,true);
        // Normals must also stay transverse to the fiber for a cross direction to exist.
        for(int c=0;c<cols;++c)if(std::abs(normal[c].dot(tangent[c]))>std::cos(20*CV_PI/180))badNormal[c]=1;
        interpolateBad(normal,badNormal,true);
    }
    std::vector<cv::Vec3d> cross(cols);std::vector<char> badCross(cols,0);
    for(int c=0;c<cols;++c){cross[c]=detail::unit(normal[c].cross(tangent[c]));badCross[c]=cv::norm(cross[c])<.5;}
    interpolateBad(cross,badCross,true);
    // March each side outward, following the normal field so the ribbon bends
    // with the sheet. A step that meets an invalid or strongly disagreeing
    // normal keeps the previous in-sheet direction.
    const int stepsPerRow=std::max(1,int(std::lround(spacing/o.step)));const double step=spacing/stepsPerRow;
    // The sampler's own sheet score unless the caller supplied (or disabled) one.
    std::function<std::optional<double>(const cv::Vec3d&)> sheetScore=o.sheetScore;
    if(!sheetScore&&sampler.sampleSheetScore(center[0]))sheetScore=[&sampler](const cv::Vec3d& q){return sampler.sampleSheetScore(q);};
    std::vector<cv::Vec3d> xyz(size_t(rows)*cols);
    for(int c=0;c<cols;++c)xyz[size_t(middle)*cols+c]=center[c];
    for(int c=0;c<cols;++c) {
        detail::cancelled(cancel);
        for(int side:{-1,1}) {
            cv::Vec3d p=center[c],direction=cross[c]*double(side),n=normal[c];
            for(int r=1;r<=middle;++r) {
                for(int k=0;k<stepsPerRow;++k) {
                    const auto sample=sampler.sampleNormal(p);
                    if(sample.valid&&cv::norm(sample.normal)>1e-6) {
                        auto local=detail::unit(sample.normal);if(local.dot(n)<0)local=-local;
                        if(local.dot(n)>std::cos(o.followAngle*CV_PI/180))n=local;
                    }
                    auto d=detail::unit(direction-direction.dot(n)*n);
                    if(cv::norm(d)<.5||d.dot(direction)<.5)d=direction;
                    direction=d;p+=direction*step;
                    if(sheetScore) {
                        // Re-centre on the sheet: climb the score along the normal to its
                        // nearest clear peak, refined by a parabola through the neighbours.
                        const auto here=sheetScore(p);
                        if(here) {
                            double bestT=0,bestV=*here;std::vector<std::pair<double,double>> samples{{0,*here}};
                            for(double t=o.snapStep;t<=o.snapRange+1e-9;t+=o.snapStep)for(double sign:{-1.,1.}) {
                                const auto v=sheetScore(p+n*(sign*t));if(!v)continue;samples.emplace_back(sign*t,*v);
                                if(*v>bestV||(*v==bestV&&std::abs(sign*t)<std::abs(bestT))){bestV=*v;bestT=sign*t;}
                            }
                            if(bestV-*here>=o.snapMinContrast||bestT==0) {
                                double refined=bestT;
                                auto valueAt=[&](double t)->std::optional<double>{for(const auto& [tt,v]:samples)if(std::abs(tt-t)<1e-9)return v;return std::nullopt;};
                                const auto lo=valueAt(bestT-o.snapStep),hi=valueAt(bestT+o.snapStep);
                                if(lo&&hi){const double denom=*lo-2*bestV+*hi;if(denom<-1e-9)refined=bestT+.5*o.snapStep*(*lo-*hi)/denom;}
                                p+=n*refined;
                            }
                        }
                    }
                }
                xyz[size_t(middle+side*r)*cols+c]=p;
            }
        }
    }
    // Bends: the curvature vector (change of tangent) points to the inside of
    // the bend, where a fixed-width offset ribbon folds once the offset exceeds
    // the bend radius. Rows on that side contract toward the radius instead.
    std::vector<double> radius(cols,std::numeric_limits<double>::infinity());std::vector<cv::Vec3d> curvatureDir(cols,cv::Vec3d{0,0,0});
    for(int c=1;c+1<cols;++c) {
        const auto dt=tangent[c+1]-tangent[c-1];const double angle=std::asin(std::clamp(cv::norm(tangent[c+1].cross(tangent[c-1])),0.,1.));
        if(angle>1e-6){radius[c]=(columns[c+1]-columns[c-1])/angle;curvatureDir[c]=detail::unit(dt);}
    }
    // Fold repair: a row point whose offset swings away from its neighbours',
    // runs backwards along the fiber or collapsed is interpolated; folded
    // stretches are then smoothed more widely, and inner-bend rows keep the
    // shortened length the averaging produces (capped by the bend radius).
    for(int r=0;r<rows;++r) {
        if(r==middle)continue;
        std::vector<cv::Vec3d> offset(cols);std::vector<char> bad(cols,0);
        for(int c=0;c<cols;++c)offset[c]=xyz[size_t(r)*cols+c]-center[c];
        const double length=std::abs(r-middle)*spacing;
        std::vector<double> cap(cols,length);
        for(int c=0;c<cols;++c) {
            const bool inside=curvatureDir[c].dot(detail::unit(offset[c]))>.3;
            if(inside&&std::isfinite(radius[c]))cap[c]=std::clamp(.8*radius[c],.35*length,length);
        }
        // Smooth the cap along the fiber so contraction eases in and out of the bend.
        {std::vector<cv::Vec3d> tmp(cols);for(int c=0;c<cols;++c)tmp[c]={cap[c],0,0};tmp=gaussianAlongColumns(tmp,2.);for(int c=0;c<cols;++c)cap[c]=std::min(cap[c],tmp[c][0]);}
        for(int c=0;c<cols;++c) {
            const auto o=detail::unit(offset[c]);
            const auto prev=detail::unit(offset[std::max(c-1,0)]),next=detail::unit(offset[std::min(c+1,cols-1)]);
            const bool swings=(c>0&&o.dot(prev)<std::cos(45*CV_PI/180))&&(c+1<cols&&o.dot(next)<std::cos(45*CV_PI/180));
            const bool backwards=c+1<cols&&(xyz[size_t(r)*cols+c+1]-xyz[size_t(r)*cols+c]).dot(tangent[c])<=0;
            const bool shortened=cv::norm(offset[c])<.5*length;
            bad[c]=swings||backwards||shortened;
        }
        std::vector<char> folded=bad;
        if(std::count(bad.begin(),bad.end(),1)==cols)for(int c=0;c<cols;++c){offset[c]=cross[c]*double(r-middle)*spacing;bad[c]=0;folded[c]=0;}
        interpolateBad(offset,bad,false);
        // Wider smoothing within four columns of a fold, light elsewhere.
        auto narrow=gaussianAlongColumns(offset,o.offsetSigma),wide=gaussianAlongColumns(offset,std::max(3.,o.offsetSigma));
        std::vector<char> nearFold(cols,0);
        for(int c=0;c<cols;++c)if(folded[c])for(int k=std::max(0,c-4);k<=std::min(cols-1,c+4);++k)nearFold[k]=1;
        for(int c=0;c<cols;++c) {
            const auto& sm=nearFold[c]?wide[c]:narrow[c];
            // Averaged offsets shorten where neighbouring directions disagree
            // (the inside of a bend); keep that shortening down to the cap.
            const double target=cap[c]<length-1e-9?std::clamp(cv::norm(sm),.35*length,cap[c]):length;
            xyz[size_t(r)*cols+c]=center[c]+detail::unit(sm)*target;
        }
    }
    auto node=std::make_shared<Node>();node->id=f.id;node->path=f.path;node->fiber=true;node->direction=f.direction;
    node->line=f.line;node->controls=f.controls;node->stripGeneration=kFiberStripGeneration;
    node->lineArclength=arc;node->columnArclength=columns;node->rows=rows;node->cols=cols;node->xyz=std::move(xyz);
    node->initialUv.resize(node->xyz.size());
    const auto delta=f.line.back()-f.line.front();
    cv::Vec2d along{1,0};
    if(f.direction=='V')along={0,delta[2]>=0?-1.0:1.0};
    else{const auto right=detail::unit(cv::Vec3d{0,0,-1}.cross(normal[0]));if(delta.dot(right)<0)along={-1,0};}
    const cv::Vec2d across{-along[1],along[0]};
    for(int x=0;x<cols;++x)for(int y=0;y<rows;++y)node->initialUv[size_t(y)*cols+x]=along*columns[x]+across*((y-middle)*(3*spacing/double(middle)));
    detail::triangulate(*node);
    if(node->triangles.empty())throw std::runtime_error("Fast strip has no valid quads for fiber "+f.id);
    {auto t=node->triangles.front();if(detail::cross(node->initialUv[t[1]]-node->initialUv[t[0]],node->initialUv[t[2]]-node->initialUv[t[0]])<0)for(auto& uv:node->initialUv)uv[0]*=-1;}
    return node;
}
} // namespace vc::constraints
