#pragma once

#include <cmath>
#include <iterator>
#include <optional>
#include <random>
#include <type_traits>

#include <educelab/core/utils/Math.hpp>

namespace ransac {
template <class Value, class T> struct RANSACResult {
  T error{educelab::INF<T>};
  T fitness{0.};
  std::vector<Value> inliers;
  T inlier_rmse{educelab::INF<T>};
  bool success{false};
};

template <class Xs, class FitFunc, class EvalFunc, typename T = float>
auto RANSAC(const Xs &x, const FitFunc &fit, const EvalFunc &eval,
            std::size_t nSamples, std::size_t nIters = 1000,
            const std::optional<std::mt19937::result_type> seed = std::nullopt,
            T probability = 0.99999999) {
  using ValueType = typename Xs::value_type;
  using Model = typename std::invoke_result_t<FitFunc, const Xs &>::second_type;
  using EvalResult = std::invoke_result_t<EvalFunc, const Xs &, const Model &>;

  EvalResult bestResult;
  Model bestModel;

  // Set up rng
  static std::mt19937 rng{std::random_device{}()};
  if (seed) {
    rng.seed(seed.value());
  }

  // Iterate
  std::size_t breakIter{nIters};
  for (std::size_t i = 0; i < nIters; ++i) {
    // Break early based on fitness/rmse
    if (i > breakIter) {
      break;
    }

    // Randomly sample
    Xs samples;
    std::sample(std::begin(x), std::end(x), std::back_inserter(samples),
                nSamples, rng);

    // Fit and evaluate the model
    auto [success, model] = fit(samples);
    if (not success) {
      continue;
    }
    auto result = eval(x, model);
    if (not result.success) {
      continue;
    }

    // Update our best models
    auto improvedFitness = result.fitness > bestResult.fitness;
    auto improvedRMSE = result.fitness == bestResult.fitness and
                        result.inlier_rmse > bestResult.inlier_rmse;
    if (improvedFitness or improvedRMSE) {
      bestResult = result;
      bestModel = model;
      if (result.fitness < 1.) {
        breakIter =
            std::min(static_cast<double>(nIters),
                     std::log(1. - probability) /
                         std::log(1. - std::pow(result.fitness, nSamples)));
      } else {
        break;
      }
    } // if improved
  } // for nIters

  // Finalize the best result and model
  bestResult = eval(x, bestModel);
  std::tie(bestResult.success, bestModel) = fit(bestResult.inliers);

  return std::make_pair(bestModel, bestResult);
}
} // namespace ransac