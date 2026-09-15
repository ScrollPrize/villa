#pragma once
#include "FiberTraceCli.hpp"
#include "vc/core/render/ChunkCache.hpp"
#include "vc/lasagna/ChannelSampler.hpp"
#include "utils/zarr.hpp"
#include <fstream>
#include <iostream>

inline vc::fiber_tracer::FiberTraceConfig fixedPathConfig(const nlohmann::json& request)
{
    using namespace vc::fiber_tracer;
    FiberTraceConfig config;
    std::vector<std::string> arguments = request.at("trace_args").get<std::vector<std::string>>();
    std::vector<char*> args;
    for (auto& a : arguments) args.push_back(a.data());
    for (int i=0; i<static_cast<int>(args.size()); ++i)
        if (!cli::parseTraceOption(arguments[i],i,static_cast<int>(args.size()),args.data(),config))
            throw std::invalid_argument("unknown fixed-path trace option");
    if (config.embeddingReferenceMode != "ema" || config.embeddingWeight <= 0)
        throw std::invalid_argument("fixed-path export requires positive-weight plain EMA");
    return config;
}

inline int queryFiberIntegration(int argc, char** argv)
{
    using namespace vc::fiber_tracer;
    if (argc != 3) throw std::invalid_argument("integrate-embeddings REQUEST_JSON");
    std::ifstream input(argv[2]);
    const auto requests = nlohmann::json::parse(input);
    nlohmann::json output=nlohmann::json::array();
    for (const auto& request:requests) {
        const auto config=fixedPathConfig(request);
        FiberEmbeddingMemory state;
        nlohmann::json costs=nlohmann::json::array();
        for (const auto& s:request.at("samples")) {
            const auto vector=s.at("embedding").get<std::vector<float>>();
            const double distance=s.at("distance").get<double>(), presence=s.at("presence").get<double>();
            if (state.ema.empty()) { state.ema=state.seed=vector; costs.push_back(0); continue; }
            costs.push_back(config.embeddingWeight*(config.embeddingIntegration.active() ?
                integratedFiberEmbeddingCost(vector,state,distance,config.embeddingIntegration) : fiberEmbeddingMemoryCost(vector,state,false)));
            if (config.embeddingIntegration.active())
                advanceIntegratedFiberEmbedding(state,vector,distance,config.embeddingEmaHalfLifeBaseVoxels,presence,config.embeddingIntegration);
            else advanceFiberEmbeddingMemory(state,vector,distance,config.embeddingEmaHalfLifeBaseVoxels,presence,false);
        }
        output.push_back(costs);
    }
    std::cout<<output.dump()<<'\n';
    return 0;
}

// Uses the ordinary prediction field/readiness gate and native candidate costs.
inline int queryFixedFiberPaths(int argc, char** argv)
{
    using namespace vc::fiber_tracer;
    if (argc != 3) throw std::invalid_argument("score-fixed-paths REQUEST_JSON");
    std::ifstream input(argv[2]);
    const auto request = nlohmann::json::parse(input);
    auto config=fixedPathConfig(request);
    config.traceToBaseScale = request.at("trace_to_base_scale").get<double>();
    if (!std::isfinite(config.traceToBaseScale) || config.traceToBaseScale <= 0)
        throw std::invalid_argument("invalid fixed-path scale");
    vc::render::processChunkCacheService()->configureDecodedByteCapacity(2ULL<<30);
    vc::lasagna::LasagnaDatasetOpenOptions options;
    options.workingToBaseScale = config.traceToBaseScale;
    const auto dataset = vc::lasagna::LasagnaDataset::openLocation(request.at("fiber_manifest").get<std::string>(), options);
    FiberPredictionField predictions(dataset,FiberPredictionFieldBindingMode::TraceOptions,true);
    std::vector<vc::lasagna::LasagnaChannelBinding> bindings;
    for (const auto* channel : {"presence","nx","ny"})
        bindings.push_back(vc::lasagna::bindLasagnaChannel(dataset.manifest(),channel));
    const auto embeddings = vc::lasagna::bindLasagnaEmbeddingChannels(dataset.manifest(),bindings.front());
    bindings.insert(bindings.end(),embeddings.begin(),embeddings.end());
    predictions.setReadinessGate([&](const auto& dependencies) {
        for (const auto& d : dependencies) {
            bool found = false;
            for (const auto& binding : bindings) {
                if (binding.path != d.arrayPath) continue;
                found = true;
                const std::array<size_t,3> index{d.indexZYX[0],d.indexZYX[1],d.indexZYX[2]};
                if (!std::filesystem::is_regular_file(binding.array->chunk_path(index)))
                    throw std::runtime_error("fixed path missing published prediction chunk");
                break;
            }
            if (!found) throw std::runtime_error("fixed path unknown channel dependency");
        }
    });
    const auto normals = vc::lasagna::LasagnaDataset::openLocation(request.at("normal_manifest").get<std::string>(), options);
    vc::lasagna::LasagnaNormalSampler normalSampler(normals);
    nlohmann::json output = nlohmann::json::array();
    for (const auto& path : request.at("paths")) {
        try {
            std::vector<cv::Vec3d> points;
            for (const auto& p : path) points.emplace_back(p[0].get<double>()/config.traceToBaseScale,p[1].get<double>()/config.traceToBaseScale,p[2].get<double>()/config.traceToBaseScale);
            nlohmann::json rows = nlohmann::json::array();
            for (const auto& r : scoreFiberFixedPath(predictions,points,config,&normalSampler))
                rows.push_back({{"embedding",r.embedding},{"distance",r.distanceBase},{"base_cost",r.baseCost},{"total_cost",r.totalCost},{"prediction_angle",r.predictionAngleDegrees},{"presence",r.presence}});
            output.push_back({{"status","ok"},{"samples",rows}});
        } catch (const std::exception& e) {
            output.push_back({{"status","unavailable"},{"reason",e.what()}});
        }
    }
    std::cout << output.dump() << '\n';
    return 0;
}
