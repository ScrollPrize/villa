// The robust linear least squares of the geometry steps (helirec/calibration/solve.hpp).
#include "helirec/calibration/solve.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_map>
#include <limits>
#include <Eigen/QR>
#include <Eigen/SparseQR>
#include "calibration/ceres_support.hpp"
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
        auto first = tmp.begin() + static_cast<std::ptrdiff_t>(count[i]);
        auto last = tmp.begin() + static_cast<std::ptrdiff_t>(count[i+1]);
        const auto less = [](const auto& a,const auto& b) { return a.first < b.first; };
        if (last-first > 64) std::stable_sort(first,last,less);
        else for (auto at = first; at != last; ++at) {
            auto entry = *at;
            auto pos = at;
            while (pos != first && less(entry,*(pos-1))) { *pos = *(pos-1); --pos; }
            *pos = entry;
        }
        for (std::size_t e = count[i]; e < count[i + 1]; ++e) {
            if (r.col.size() > r.start.back() && r.col.back() == tmp[e].first) r.value.back() += tmp[e].second;
            else r.col.push_back(tmp[e].first), r.value.push_back(tmp[e].second);
        }
        r.start.push_back(r.col.size());
    }
    return r;
}

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

// Equations with identical dependencies share assembly, but each scalar residual
// receives its own Huber loss. Encoding sign(r)*sqrt(rho(r*r)) lets Ceres batch the
// Jacobian while minimizing exactly the scalar robust objective. Its complete
// derivative is included; no reweighted fitting loop lives outside Ceres.
class EquationBatch final : public ceres::CostFunction {
  public:
    Eigen::MatrixXd a;
    Eigen::VectorXd y;
    double huber;
    EquationBatch(Eigen::MatrixXd coefficients, Eigen::VectorXd measured, double h)
        : a(std::move(coefficients)), y(std::move(measured)), huber(h) {
        set_num_residuals(static_cast<int>(y.size()));
        mutable_parameter_block_sizes()->assign(a.cols(),1);
    }
    bool Evaluate(double const* const* x, double* residual, double** jacobian) const override {
        Eigen::Map<Eigen::VectorXd> r(residual,y.size());
        r = y;
        for (Eigen::Index k = 0; k < a.cols(); ++k) r.noalias() -= a.col(k)*x[k][0];
        if (!r.allFinite()) return false;
        if (jacobian)
            for (Eigen::Index k = 0; k < a.cols(); ++k)
                if (jacobian[k]) Eigen::Map<Eigen::VectorXd>(jacobian[k],y.size()) = -a.col(k);
        if (huber > 0) {
            ceres::HuberLoss loss(huber);
            for (Eigen::Index i = 0; i < y.size(); ++i) {
                if (std::abs(r[i]) <= huber) continue;
                double rho[3];
                loss.Evaluate(r[i]*r[i],rho);
                const double root = std::sqrt(rho[0]), derivative = std::abs(r[i])*rho[1]/root;
                r[i] = std::copysign(root,r[i]);
                if (jacobian)
                    for (Eigen::Index k = 0; k < a.cols(); ++k) if (jacobian[k]) jacobian[k][i] *= derivative;
            }
        }
        return true;
    }
};
struct ColumnsHash {
    std::size_t operator()(const std::vector<int>& columns) const {
        std::size_t h = 0;
        for (int c : columns) h = (h ^ static_cast<std::size_t>(c))*std::size_t(0x9e3779b1);
        return h;
    }
};
struct EquationGroup {
    std::vector<int> columns;
    Eigen::MatrixXd a;
    Eigen::VectorXd y;
};
}  // namespace

LinearSolution solve_linear(const LinearProblem& problem, const SolveOptions& opt) {
    fitting::validate(opt);
    if (problem.unknowns > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        problem.y.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw Error("solve_linear: design exceeds index range");
    Rows A = by_row(problem);
    const std::size_t m = A.m, n = A.n;
    for (double v : problem.y)
        if (!std::isfinite(v)) throw Error("solve_linear: a measurement is not finite");
    if (problem.fixed.size() != problem.fixed_value.size()) throw Error("solve_linear: fixed indices need values");
    LinearSolution out;
    out.solver_version = CERES_VERSION_STRING;
    out.x.assign(n, 0);
    std::vector<bool> fixed(n, false);
    for (std::size_t k = 0; k < problem.fixed.size(); ++k) {
        const auto c = problem.fixed[k];
        if (c < 0 || static_cast<std::size_t>(c) >= n || !std::isfinite(problem.fixed_value[k]))
            throw Error("solve_linear: invalid fixed parameter");
        if (fixed[c]) throw Error("solve_linear: duplicate fixed parameter");
        fixed[c] = true;
        out.x[c] = problem.fixed_value[k];
    }
    std::vector<double> norm(n, 0), largest(n, 0), x(n, 0);
    for (std::size_t e = 0; e < A.value.size(); ++e) {
        if (!std::isfinite(A.value[e])) throw Error("solve_linear: coefficient sum is not finite");
        largest[A.col[e]] = std::max(largest[A.col[e]],std::abs(A.value[e]));
    }
    for (std::size_t e = 0; e < A.value.size(); ++e) {
        const int c = A.col[e];
        if (largest[c] > 0) { const double v = A.value[e]/largest[c]; norm[c] += v*v; }
    }
    for (std::size_t c = 0; c < n; ++c) norm[c] = largest[c]*std::sqrt(norm[c]);
    std::vector<int> active(n, -1);
    for (std::size_t k = 0; k < n; ++k) {
        if (norm[k] == 0) { ++out.unused; fixed[k] = true; norm[k] = 1; }
        if (!std::isfinite(norm[k])) throw Error("solve_linear: column norm overflow");
        if (!fixed[k]) active[k] = out.active++;
        x[k] = out.x[k] * norm[k];
        if (!std::isfinite(x[k])) throw Error("solve_linear: fixed parameter overflow");
    }
    std::unordered_map<std::vector<int>,std::size_t,ColumnsHash> lookup;
    std::vector<std::pair<std::vector<int>,std::vector<std::size_t>>> grouped;
    std::vector<int> columns;
    for (std::size_t i = 0; i < m; ++i) {
        columns.clear();
        for (std::size_t e = A.start[i]; e < A.start[i+1]; ++e) {
            A.value[e] /= norm[A.col[e]];
            if (A.value[e] != 0) columns.push_back(A.col[e]);
        }
        auto it = lookup.find(columns);
        if (it == lookup.end()) {
            it = lookup.emplace(columns,grouped.size()).first;
            grouped.push_back({columns,{}});
        }
        grouped[it->second].second.push_back(i);
    }
    std::vector<EquationGroup> groups;
    using Sparse = Eigen::SparseMatrix<double>;
    std::vector<Eigen::Triplet<double>> entries;
    int rank_rows = 0;
    for (const auto& [columns,rows] : grouped) {
        EquationGroup group;
        group.columns = columns;
        group.a.resize(rows.size(),columns.size());
        group.y.resize(rows.size());
        for (std::size_t i = 0; i < rows.size(); ++i) {
            group.y[i] = problem.y[rows[i]];
            int k = 0;
            for (std::size_t e = A.start[rows[i]]; e < A.start[rows[i]+1]; ++e)
                if (A.value[e] != 0) group.a(i,k++) = A.value[e];
        }
        std::vector<int> cols;
        for (int c : columns) if (active[c] >= 0) cols.push_back(c);
        if (!cols.empty()) {
            Eigen::MatrixXd reduced(rows.size(),cols.size());
            int k = 0;
            for (std::size_t j = 0; j < columns.size(); ++j)
                if (active[columns[j]] >= 0) reduced.col(k++) = group.a.col(j);
            // Orthogonal row compression preserves the design's singular values.
            // Rank analysis needs R alone, not a huge Q for all repeated observations.
            Eigen::HouseholderQR<Eigen::MatrixXd> qr(reduced);
            const int nr = std::min(reduced.rows(),reduced.cols());
            for (int i = 0; i < nr; ++i)
                for (std::size_t j = i; j < cols.size(); ++j)
                    if (qr.matrixQR()(i,j) != 0) entries.emplace_back(rank_rows+i,active[cols[j]],qr.matrixQR()(i,j));
            rank_rows += nr;
        }
        groups.push_back(std::move(group));
    }
    if (out.active) {
        Sparse design(rank_rows,out.active);
        design.setFromTriplets(entries.begin(),entries.end());
        if (out.active <= 64) {
            Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr{Eigen::MatrixXd(design)};
            qr.setThreshold(opt.rcond);
            out.rank = static_cast<int>(qr.rank());
        } else {
            Eigen::SparseQR<Sparse,Eigen::COLAMDOrdering<int>> qr;
            qr.setPivotThreshold(opt.rcond);
            qr.compute(design);
            if (qr.info() != Eigen::Success) throw Error("solve_linear: rank analysis failed");
            out.rank = static_cast<int>(qr.rank());
        }
    }
    if (!m || out.rank != out.active) {
        out.termination = !m ? "INSUFFICIENT_SUPPORT" : "RANK_DEFICIENT";
        out.message = "solve_linear: rank " + std::to_string(out.rank) + " of " + std::to_string(out.active) +
                      " active parameters; fix unobservable parameters explicitly";
    } else {
        ceres::Problem fit;
        for (std::size_t k = 0; k < n; ++k) {
            fit.AddParameterBlock(&x[k], 1);
            if (fixed[k]) fit.SetParameterBlockConstant(&x[k]);
        }
        std::vector<EquationBatch*> batches;
        for (auto& group : groups) {
            std::vector<double*> blocks;
            for (int c : group.columns) blocks.push_back(&x[c]);
            auto* batch = new EquationBatch(std::move(group.a),std::move(group.y),0);
            batches.push_back(batch);
            fit.AddResidualBlock(batch,nullptr,blocks);
        }
        ceres::Solver::Summary s;
        auto options = fitting::options(out.active, opt);
        options.initial_trust_region_radius = 1e16;
        options.max_trust_region_radius = 1e20;
        FitSummary initialization;
        if (opt.huber > 0 && opt.max_iterations > 3) {
            auto initial_options = options;
            initial_options.max_num_iterations = 2;
            ceres::Solver::Summary initial;
            ceres::Solve(initial_options,&fit,&initial);
            initialization = fitting::summary(initial);
            options.max_num_iterations -= initialization.iterations;
        }
        for (auto* batch : batches) batch->huber = opt.huber;
        ceres::Solve(options, &fit, &s);
        static_cast<FitSummary&>(out) = fitting::summary(s);
        out.iterations += initialization.iterations;
        out.evaluations += initialization.evaluations;
    }
    out.residual = problem.y;
    out.weight.assign(m, 1);
    for (std::size_t k = 0; k < n; ++k) out.x[k] = x[k] / norm[k];
    for (std::size_t k = 0; k < problem.fixed.size(); ++k) out.x[problem.fixed[k]] = problem.fixed_value[k];
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t e = A.start[i]; e < A.start[i+1]; ++e) out.residual[i] -= A.value[e] * x[A.col[e]];
        if (opt.huber > 0 && std::abs(out.residual[i]) > opt.huber)
            out.weight[i] = std::sqrt(opt.huber / std::abs(out.residual[i]));
    }
    out.spread = robust_spread(out.residual);
    return out;
}

LineFit fit_line(const std::vector<double>& x, const std::vector<double>& y, const SolveOptions& options) {
    if (x.size() != y.size()) throw Error("fit_line: x and y sizes differ");
    LineFit out;
    out.n = x.size();
    out.solver_version = CERES_VERSION_STRING;
    out.slope = out.intercept = out.rms = out.max_abs = std::numeric_limits<double>::quiet_NaN();
    if (x.size() < 2) { out.termination = "INSUFFICIENT_SUPPORT"; out.message = "fit_line: need at least two points"; return out; }
    long double sx = 0, sy = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (!std::isfinite(x[i]) || !std::isfinite(y[i])) throw Error("fit_line: nonfinite input");
        sx += x[i]; sy += y[i];
    }
    const double mx = static_cast<double>(sx / x.size()), my = static_cast<double>(sy / x.size());
    double scale = 0;
    for (double v : x) scale = std::max(scale, std::abs(v - mx));
    if (scale == 0) { out.termination = "RANK_DEFICIENT"; out.message = "fit_line: predictor is constant"; return out; }
    LinearProblem p;
    p.unknowns = 2;
    for (std::size_t i = 0; i < x.size(); ++i) {
        p.row.insert(p.row.end(), {static_cast<std::int64_t>(i), static_cast<std::int64_t>(i)});
        p.col.insert(p.col.end(), {0, 1});
        p.value.insert(p.value.end(), {1, (x[i] - mx) / scale});
        p.y.push_back(y[i] - my);
    }
    auto fit = solve_linear(p, options);
    static_cast<FitSummary&>(out) = fit;
    if (!fit.usable) return out;
    out.slope = fit.x[1] / scale;
    out.intercept = static_cast<double>(static_cast<long double>(my) + fit.x[0] - static_cast<long double>(out.slope) * mx);
    double ss = 0;
    out.max_abs = 0;
    for (double r : fit.residual) { ss += r*r; out.max_abs = std::max(out.max_abs, std::abs(r)); }
    out.rms = std::sqrt(ss / x.size());
    return out;
}
}  // namespace helirec::calibration
