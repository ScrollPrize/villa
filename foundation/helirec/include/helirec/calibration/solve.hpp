#pragma once
// The solver of the geometry steps: a linear model fitted robustly.
//
// A calibration is many equations, each a measurement that is a line in a few of many unknowns (a comparison of
// two blocks is a line in the poses, the map's coefficients and the knots of the courses its two blocks touch):
//
//     y_i = sum_k A_ik x_k + noise          i over the equations, A sparse
//
// solve_linear finds x by least squares under a Huber loss, as a weighted least squares repeated with the weights
// sqrt(min(1, k / |residual|)) of the solve before. Everything is helirec's own arithmetic (no outside solver):
// the columns are scaled to unit length (the unknowns of a calibration run from 1e-7 to thousands, and the
// unscaled problem is what defeats a general optimiser), the normal equations are formed in double and factored
// by a Cholesky decomposition with pivoting, and each solve is refined against the equations themselves.
// An unknown that the equations do not determine (a column of zeros, or one that is a combination of others to
// within `rcond`) comes out zero and is counted out of `rank`: a model should pin what it cannot see rather than
// lean on that.
//
// The result does not depend on the number of threads.
#include <cstddef>
#include <cstdint>
#include <vector>

#include "helirec/common.hpp"

namespace helirec::calibration {

struct LinearProblem {
    std::size_t unknowns = 0;
    // the entries of A in any order; entries with the same row and column add
    std::vector<std::int64_t> row, col;
    std::vector<double> value;
    std::vector<double> y;  // one per equation: the number of equations is y.size()
};

struct SolveOptions {
    // Threshold of the Huber loss, in the equations' units: residuals beyond it count linearly. 0: least squares.
    double huber = 0;
    // The threshold is `huber` times the residuals' own robust spread (1.4826 times the median distance from
    // their median) of the solve before, so that it is not a number in the equations' units.
    bool relative = false;
    int iterations = 8;    // weighted solves (the first with all weights 1)
    double rcond = 1e-12;  // an unknown is determined if its pivot exceeds this times the largest
};

struct LinearSolution {
    std::vector<double> x;         // the unknowns
    std::vector<double> residual;  // y - A x per equation
    std::vector<double> weight;    // of the last solve, per equation (1: inside the threshold)
    double spread = 0;             // the residuals' robust spread after the last solve
    int rank = 0;                  // unknowns determined in the last solve
};

LinearSolution solve_linear(const LinearProblem& problem, const SolveOptions& options = {});

}  // namespace helirec::calibration
