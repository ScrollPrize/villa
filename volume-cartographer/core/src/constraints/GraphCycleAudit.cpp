#include "GraphInternal.hpp"
#include "vc/fiber_tracer/FiberWindingGeometry.hpp"
#include <numeric>
#include <set>
#include <sstream>

namespace vc::constraints {
namespace {
struct Forest {
    std::vector<size_t> roots;
    explicit Forest(size_t n) : roots(n) { std::iota(roots.begin(), roots.end(), 0); }
    size_t root(size_t i) { return roots[i] == i ? i : roots[i] = root(roots[i]); }
    bool join(size_t a, size_t b) {
        a = root(a); b = root(b);
        if (a == b) return false;
        roots[std::max(a,b)] = std::min(a,b);
        return true;
    }
};
}

CycleAudit auditFiberCycles(const Catalog& catalog, const std::vector<cv::Vec3d>& centers,
                           double minimumRadius, const std::atomic_bool* cancel)
{
    namespace wg = fiber_tracer::winding_geometry;
    CycleAudit out;
    std::map<std::string,size_t> indices;
    for (const auto& [id,f] : catalog.fibers) indices.emplace(id, indices.size());
    Forest topology(indices.size());
    for (const auto& link : catalog.links) {
        if (!indices.contains(link.a) || !indices.contains(link.b)) continue;
        if (!topology.join(indices.at(link.a), indices.at(link.b))) ++out.cycles;
    }
    out.uncertainCycles = out.cycles;
    if (centers.empty()) {
        out.summary = "Fiber cycle check unavailable: no umbilicus centerline (" + std::to_string(out.cycles) + " independent cycles).";
        return out;
    }
    if (!std::isfinite(minimumRadius) || minimumRadius <= 0)
        throw std::runtime_error("Cycle audit minimum radius must be positive and finite");
    for (const auto& p : centers) for (double v : {p[0],p[1],p[2]})
        if (!std::isfinite(v)) throw std::runtime_error("Nonfinite umbilicus center");
    const auto umbilicus = wg::interpolateUmbilicus(centers);
    for (size_t i = 1; i < umbilicus.z.size(); ++i)
        if (umbilicus.z[i] <= umbilicus.z[i-1]) throw std::runtime_error("Umbilicus centers must have distinct z coordinates");

    struct Trace { wg::PolarTrace polar; bool usable = true; bool reliable = false; };
    std::vector<Trace> traces(indices.size());
    std::set<std::string> linkedFibers;
    for(const auto& link:catalog.links){linkedFibers.insert(link.a);linkedFibers.insert(link.b);}
    size_t interpolatedFibers=0, extrapolatedFibers=0;
    for (const auto& [id,f] : catalog.fibers) {
        detail::cancelled(cancel);
        auto& trace = traces[indices.at(id)];
        trace.polar = wg::polarTrace(f.line, umbilicus);
        trace.usable = f.line.size() >= 2;
        trace.reliable = f.fullyTraced;
        bool extrapolated=false;
        for (size_t i = 0; i < f.line.size(); ++i) {
            if (!std::isfinite(trace.polar.theta[i]) || !std::isfinite(trace.polar.radius[i]) ||
                trace.polar.radius[i] < minimumRadius) trace.usable = false;
            if (i && std::abs(trace.polar.theta[i] - trace.polar.theta[i-1]) > .25 * wg::kTwoPi) trace.usable = false;
            // Extrapolating the scroll center beyond its measured z range is
            // useful for drawing, but is not evidence of an annotation error.
            if (f.line[i][2] < umbilicus.z.front() || f.line[i][2] > umbilicus.z.back()) {trace.reliable = false;extrapolated=true;}
        }
        if(linkedFibers.contains(id)) {
            if(!f.fullyTraced)++interpolatedFibers;
            if(extrapolated)++extrapolatedFibers;
        }
    }
    struct Edge { size_t a,b,link; long long shift; bool reliable; };
    std::vector<Edge> edges;
    for (size_t i = 0; i < catalog.links.size(); ++i) {
        const auto& l = catalog.links[i];
        if (!indices.contains(l.a) || !indices.contains(l.b)) { ++out.skippedLinks; continue; }
        const auto a = indices.at(l.a), b = indices.at(l.b);
        const auto& fa = catalog.fibers.at(l.a); const auto& fb = catalog.fibers.at(l.b);
        if (l.pending || !traces[a].usable || !traces[b].usable ||
            l.controlA >= fa.controls.size() || l.controlB >= fb.controls.size() ||
            fa.controls[l.controlA] >= fa.line.size() || fb.controls[l.controlB] >= fb.line.size()) {
            ++out.skippedLinks; continue;
        }
        const double turns = (traces[a].polar.theta[fa.controls[l.controlA]] -
                              traces[b].polar.theta[fb.controls[l.controlB]]) / wg::kTwoPi;
        // Links preserve the endpoints' angular offset. Only their whole-turn
        // gauge difference is constrained; endpoints need not coincide in XYZ.
        const auto shift = static_cast<long long>(std::nearbyint(turns));
        if (std::abs(turns - shift) >= .25) { ++out.skippedLinks; continue; }
        edges.push_back({a,b,i,shift,traces[a].reliable && traces[b].reliable});
    }
    // Build the reliable forest first, so uncertain geometry cannot obscure
    // a cycle that could have been checked entirely using traced geometry.
    std::stable_sort(edges.begin(), edges.end(), [](const auto& a,const auto& b){return a.reliable > b.reliable;});
    Forest forest(indices.size());
    std::vector<std::vector<size_t>> adjacent(indices.size());
    std::vector<size_t> closures;
    for (size_t i = 0; i < edges.size(); ++i) {
        const auto& e = edges[i];
        if (forest.join(e.a,e.b)) { adjacent[e.a].push_back(i); adjacent[e.b].push_back(i); }
        else closures.push_back(i);
    }
    for (auto closing : closures) {
        detail::cancelled(cancel);
        ++out.evaluatedCycles;
        const auto& e = edges[closing];
        const size_t none = edges.size();
        std::vector<size_t> parent(indices.size(), none), queue{e.a};
        std::vector<bool> seen(indices.size(),false); seen[e.a] = true;
        for (size_t i = 0; i < queue.size() && !seen[e.b]; ++i) {
            for (auto index : adjacent[queue[i]]) {
                const auto& edge = edges[index];
                auto next = edge.a == queue[i] ? edge.b : edge.a;
                if (seen[next]) continue;
                seen[next] = true; parent[next] = index; queue.push_back(next);
            }
        }
        CycleIssue issue; issue.turns = -e.shift; issue.links.push_back(e.link); issue.reliable = e.reliable;
        for (size_t node = e.b; node != e.a;) {
            const auto& edge = edges[parent[node]];
            issue.turns += node == edge.b ? edge.shift : -edge.shift;
            issue.links.push_back(edge.link); issue.reliable &= edge.reliable;
            node = node == edge.b ? edge.a : edge.b;
        }
        if (issue.reliable) ++out.checkedCycles;
        if (!issue.turns) continue;
        std::ostringstream message;
        message << "Nonzero winding closure: " << issue.turns << " turn(s)";
        if (!issue.reliable) message << " (uncertain geometry; not a confirmed annotation error)";
        for (auto linkIndex : issue.links) {
            const auto& l = catalog.links[linkIndex];
            message << "\n  " << l.a << " [control " << l.controlA << "] <-> "
                    << l.b << " [control " << l.controlB << "]";
        }
        out.details.push_back(message.str()); out.issues.push_back(std::move(issue));
    }
    out.available = true;
    out.uncertainCycles = out.cycles - out.checkedCycles;
    if (!out.issues.empty())
        out.summary = "Fiber cycle check: " + std::to_string(out.issues.size()) + " nonzero winding closures found.";
    else if (out.cycles && !out.evaluatedCycles)
        out.summary = "Fiber cycle check: no cycles could be evaluated; see details.";
    else
        out.summary = "Fiber cycle check: no inconsistencies found (" + std::to_string(out.evaluatedCycles) +
            "/" + std::to_string(out.cycles) + " independent cycles evaluated).";
    out.details.push_back(std::to_string(out.checkedCycles) + "/" + std::to_string(out.cycles) +
        " independent cycles supported entirely by traced geometry.");
    if (out.uncertainCycles || out.skippedLinks)
        out.details.push_back(std::to_string(out.uncertainCycles) + " cycles lack reliable coverage; " + std::to_string(out.skippedLinks) +
            " links excluded (pending, invalid, near the centerline, undersampled, or angularly ambiguous). Interpolated fibers and extrapolated centerlines are uncertain.");
    out.details.push_back("Among linked fibers: " + std::to_string(interpolatedFibers) + " contain untraced/interpolated spans; " +
        std::to_string(extrapolatedFibers) + " extend outside the measured centerline z range.");
    out.details.push_back("Scope: original fiber folder, before fitting; all components. Control indices are zero-based. "
        "This checks whole-turn winding consistency, not all metric or rotational holonomy. "
        "A nonzero closure needs review or a chart cut; the cycle alone does not identify a bad link. "
        "Curvature and noncoincident linked endpoints can produce geometric residuals without annotation errors.");
    return out;
}
} // namespace vc::constraints
