// The robust linear least squares of the geometry steps (helirec/calibration/solve.hpp).
#include "helirec/calibration/solve.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>

namespace helirec::calibration {

namespace {

// the equations by row: entries of one row are contiguous, sorted by column, repeated entries added
struct Rows {
    std::size_t m = 0, n = 0;
    std::vector<std::size_t> start;  // m + 1
    std::vector<std::int32_t> col;
    std::vector<double> value;
};

Rows by_row(const LinearProblem& p) {
    const std::size_t m = p.y.size(), n = p.unknowns, nnz = p.value.size();
    if (p.row.size() != nnz || p.col.size() != nnz) throw Error("solve_linear: one row, one column and one value per entry");
    Rows r;
    r.m = m, r.n = n;
    std::vector<std::size_t> count(m + 1, 0);
    for (std::size_t e = 0; e < nnz; ++e) {
        if (p.row[e] < 0 || static_cast<std::size_t>(p.row[e]) >= m) throw Error("solve_linear: an entry's row is not one of the equations");
        if (p.col[e] < 0 || static_cast<std::size_t>(p.col[e]) >= n) throw Error("solve_linear: an entry's column is not one of the unknowns");
        if (!std::isfinite(p.value[e])) throw Error("solve_linear: an entry is not a number");
        ++count[static_cast<std::size_t>(p.row[e]) + 1];
    }
    for (std::size_t i = 0; i < m; ++i) count[i + 1] += count[i];
    std::vector<std::pair<std::int32_t, double>> tmp(nnz);
    std::vector<std::size_t> at(count.begin(), count.end() - 1);
    for (std::size_t e = 0; e < nnz; ++e) tmp[at[static_cast<std::size_t>(p.row[e])]++] = {static_cast<std::int32_t>(p.col[e]), p.value[e]};
    r.start.assign(1, 0);
    r.col.reserve(nnz);
    r.value.reserve(nnz);
    for (std::size_t i = 0; i < m; ++i) {
        // a stable order within the row: the sums below are formed in it whatever order the entries came in
        std::stable_sort(tmp.begin() + static_cast<std::ptrdiff_t>(count[i]), tmp.begin() + static_cast<std::ptrdiff_t>(count[i + 1]),
                         [](const auto& a, const auto& b) { return a.first < b.first; });
        for (std::size_t e = count[i]; e < count[i + 1]; ++e) {
            if (r.col.size() > r.start.back() && r.col.back() == tmp[e].first) r.value.back() += tmp[e].second;
            else r.col.push_back(tmp[e].first), r.value.push_back(tmp[e].second);
        }
        r.start.push_back(r.col.size());
    }
    return r;
}

// Cholesky with pivoting of a symmetric positive semi-definite matrix (n x n, full storage; destroyed): P^T N P =
// L L^T for the first `rank` pivots, the rest left out when the largest remaining pivot falls to `tolerance`.
struct Factor {
    std::size_t n = 0;
    int rank = 0;
    std::vector<double> L;         // n x n, lower triangle of the permuted matrix
    std::vector<std::size_t> perm; // position k of the factor is unknown perm[k]

    void make(std::vector<double>& N, std::size_t n_, double rcond) {
        n = n_;
        perm.resize(n);
        std::iota(perm.begin(), perm.end(), std::size_t(0));
        L.assign(n * n, 0.0);
        double largest = 0;
        for (std::size_t k = 0; k < n; ++k) largest = std::max(largest, N[k * n + k]);
        const double tolerance = rcond * largest;
        rank = 0;
        for (std::size_t k = 0; k < n; ++k) {
            std::size_t p = k;
            for (std::size_t j = k + 1; j < n; ++j)
                if (N[j * n + j] > N[p * n + p]) p = j;
            if (!(N[p * n + p] > tolerance) || !(N[p * n + p] > 0)) break;
            if (p != k) {  // rows and columns k and p of the symmetric matrix, and what is factored of them
                for (std::size_t j = 0; j < n; ++j) std::swap(N[k * n + j], N[p * n + j]);
                for (std::size_t j = 0; j < n; ++j) std::swap(N[j * n + k], N[j * n + p]);
                for (std::size_t j = 0; j < k; ++j) std::swap(L[k * n + j], L[p * n + j]);
                std::swap(perm[k], perm[p]);
            }
            const double d = std::sqrt(N[k * n + k]);
            L[k * n + k] = d;
            for (std::size_t i = k + 1; i < n; ++i) L[i * n + k] = N[i * n + k] / d;
            const std::ptrdiff_t first = static_cast<std::ptrdiff_t>(k) + 1, last = static_cast<std::ptrdiff_t>(n);
            // what is left, both triangles (the swaps read both; the two halves get the same products)
#pragma omp parallel for schedule(static) if (n - k > 256)
            for (std::ptrdiff_t i = first; i < last; ++i) {
                const double lik = L[static_cast<std::size_t>(i) * n + k];
                double* row = N.data() + static_cast<std::size_t>(i) * n;
                for (std::size_t j = k + 1; j < n; ++j) row[j] -= lik * L[j * n + k];
            }
            ++rank;
        }
    }

    // x with N x = g on the determined unknowns, zero on the others
    std::vector<double> solve(const std::vector<double>& g) const {
        const std::size_t r = static_cast<std::size_t>(rank);
        std::vector<double> z(r), x(n, 0.0);
        for (std::size_t k = 0; k < r; ++k) {
            double s = g[perm[k]];
            for (std::size_t j = 0; j < k; ++j) s -= L[k * n + j] * z[j];
            z[k] = s / L[k * n + k];
        }
        for (std::size_t k = r; k-- > 0;) {
            double s = z[k];
            for (std::size_t j = k + 1; j < r; ++j) s -= L[j * n + k] * z[j];
            z[k] = s / L[k * n + k];
        }
        for (std::size_t k = 0; k < r; ++k) x[perm[k]] = z[k];
        return x;
    }
};

double robust_spread(std::vector<double> v) {
    if (v.empty()) return 0;
    const auto median = [](std::vector<double>& a) {
        const std::size_t h = a.size() / 2;
        std::nth_element(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(h), a.end());
        double m = a[h];
        if (a.size() % 2 == 0) m = 0.5 * (m + *std::max_element(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(h)));
        return m;
    };
    const double centre = median(v);
    for (double& x : v) x = std::fabs(x - centre);
    return 1.4826 * median(v);
}

}  // namespace

LinearSolution solve_linear(const LinearProblem& problem, const SolveOptions& opt) {
    if (opt.iterations < 1) throw Error("solve_linear: at least one solve");
    if (!(opt.huber >= 0)) throw Error("solve_linear: the threshold is not negative");
    Rows A = by_row(problem);
    const std::size_t m = A.m, n = A.n;
    for (double v : problem.y)
        if (!std::isfinite(v)) throw Error("solve_linear: a measurement is not a number");
    LinearSolution out;
    out.x.assign(n, 0.0);
    out.residual = problem.y;
    out.weight.assign(m, 1.0);
    if (!n || !m) return out;
    // the columns to unit length
    std::vector<double> norm(n, 0.0);
    for (std::size_t e = 0; e < A.value.size(); ++e) norm[static_cast<std::size_t>(A.col[e])] += A.value[e] * A.value[e];
    for (double& v : norm) v = v > 0 ? std::sqrt(v) : 1.0;
    for (std::size_t e = 0; e < A.value.size(); ++e) A.value[e] /= norm[static_cast<std::size_t>(A.col[e])];

    std::vector<double> x(n, 0.0), w(m, 1.0), N, g(n), r(m);
    Factor factor;
    const auto residuals = [&](const std::vector<double>& at) {
        for (std::size_t i = 0; i < m; ++i) {
            double s = problem.y[i];
            for (std::size_t e = A.start[i]; e < A.start[i + 1]; ++e) s -= A.value[e] * at[static_cast<std::size_t>(A.col[e])];
            r[i] = s;
        }
    };
    const auto gradient = [&](const std::vector<double>& of) {  // A^T W^2 of
        std::fill(g.begin(), g.end(), 0.0);
        for (std::size_t i = 0; i < m; ++i) {
            const double s = w[i] * w[i] * of[i];
            for (std::size_t e = A.start[i]; e < A.start[i + 1]; ++e) g[static_cast<std::size_t>(A.col[e])] += A.value[e] * s;
        }
    };
    const int solves = opt.huber > 0 ? opt.iterations : 1;
    for (int it = 0; it < solves; ++it) {
        // the normal equations of the weighted problem, an equation at a time (a few entries each)
        N.assign(n * n, 0.0);
        for (std::size_t i = 0; i < m; ++i) {
            const double w2 = w[i] * w[i];
            if (w2 == 0) continue;
            for (std::size_t a = A.start[i]; a < A.start[i + 1]; ++a) {
                const double va = w2 * A.value[a];
                double* row = N.data() + static_cast<std::size_t>(A.col[a]) * n;
                for (std::size_t b = A.start[i]; b <= a; ++b) row[static_cast<std::size_t>(A.col[b])] += va * A.value[b];
            }
        }
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < i; ++j) N[j * n + i] = N[i * n + j];
        factor.make(N, n, opt.rcond);
        gradient(problem.y);
        x = factor.solve(g);
        // refined against the equations themselves: the normal equations lose half the digits, and a model with
        // weakly held directions (a course against a constant) shows it
        for (int pass = 0; pass < 6; ++pass) {
            residuals(x);
            gradient(r);
            const std::vector<double> dx = factor.solve(g);
            double change = 0, size = 0;
            for (std::size_t k = 0; k < n; ++k) {
                x[k] += dx[k];
                change = std::max(change, std::fabs(dx[k]));
                size = std::max(size, std::fabs(x[k]));
            }
            if (change <= 1e-13 * size) break;
        }
        residuals(x);
        out.rank = factor.rank;
        out.weight = w;
        out.spread = robust_spread(r);
        if (it + 1 == solves) break;
        const double k = opt.relative ? opt.huber * out.spread : opt.huber;
        for (std::size_t i = 0; i < m; ++i) w[i] = std::sqrt(std::min(1.0, k / std::max(std::fabs(r[i]), 1e-12)));
    }
    for (std::size_t k = 0; k < n; ++k) out.x[k] = x[k] / norm[k];
    out.residual = r;
    return out;
}

}  // namespace helirec::calibration
