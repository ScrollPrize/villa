#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include <utility>

namespace vc::fiber_tracer::detail {
// Deterministic memory: remember the last active label through disabled spans.
// Each visible assignment has exactly one memory path (no latent entropy).
// Labels are [0,labelCount); -1 denotes disabled. Costs are already normalized.
inline double sequenceLogAdd(double a, double b)
{
    if (!std::isfinite(a)) return b;
    if (!std::isfinite(b)) return a;
    return std::max(a,b)+std::log1p(std::exp(-std::abs(a-b)));
}

inline std::vector<std::vector<double>> windingSequenceMessages(
    const std::vector<std::vector<int>>& labels,
    const std::vector<std::vector<double>>& cavity,
    std::size_t labelCount, double breakCost, double openingCost)
{
    const double neg=-std::numeric_limits<double>::infinity();
    if(labels.size()!=cavity.size() || std::isnan(breakCost) || breakCost<0. ||
       !std::isfinite(openingCost) || openingCost<0.)
        throw std::invalid_argument("Invalid winding sequence factor");
    const std::size_t n=labels.size(), states=2*(labelCount+1);
    // memory/2 == labelCount means no active label seen yet; memory%2 is Defect.
    auto transition=[&](std::size_t memory,int label) {
        const auto previous=memory/2;
        if(label<0) return std::pair{previous*2+1, memory%2 ? 0. : openingCost};
        const bool change=previous!=labelCount && previous!=static_cast<std::size_t>(label);
        return std::pair{static_cast<std::size_t>(label)*2, change?breakCost:0.};
    };
    for(std::size_t i=0;i<n;++i) {
        if(labels[i].size()!=cavity[i].size()) throw std::invalid_argument("Sequence state shape");
        for(auto label:labels[i]) if(label < -1 || label>=static_cast<int>(labelCount))
            throw std::invalid_argument("Sequence label outside alphabet");
    }
    std::vector<std::vector<double>> forward(n+1,std::vector<double>(states,neg));
    forward[0][labelCount*2]=0.;
    for(std::size_t i=0;i<n;++i)
        for(std::size_t m=0;m<states;++m) if(std::isfinite(forward[i][m]))
            for(std::size_t s=0;s<labels[i].size();++s) {
                auto [next,cost]=transition(m,labels[i][s]);
                forward[i+1][next]=sequenceLogAdd(forward[i+1][next],
                    forward[i][m]+cavity[i][s]-cost);
            }
    std::vector<double> back(states,0.);
    std::vector<std::vector<double>> result(n);
    for(std::size_t i=n;i-->0;) {
        std::vector<double> previous(states,neg);
        result[i].assign(labels[i].size(),neg);
        for(std::size_t m=0;m<states;++m)
            for(std::size_t s=0;s<labels[i].size();++s) {
                auto [next,cost]=transition(m,labels[i][s]);
                const double suffix=back[next]-cost;
                previous[m]=sequenceLogAdd(previous[m],cavity[i][s]+suffix);
                result[i][s]=sequenceLogAdd(result[i][s],forward[i][m]+suffix);
            }
        back=std::move(previous);
    }
    return result;
}

inline double windingSequenceCost(const std::vector<int>& labels,
                                  double breakCost,double openingCost)
{
    int last=-1; bool disabled=false; double cost=0.;
    for(int label:labels) {
        if(label<0) { if(!disabled)cost+=openingCost; disabled=true; }
        else { if(last>=0 && last!=label)cost+=breakCost; last=label; disabled=false; }
    }
    return cost;
}
}
