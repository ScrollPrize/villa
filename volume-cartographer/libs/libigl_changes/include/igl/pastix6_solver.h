// PaStiX 6.x SPD solver shim with an Eigen-compatible compute()/solve()
// interface. Mirrors the slice of Eigen::CholmodSupernodalLLT that SLIM
// actually uses (analyzePattern + factorize + solve) so we can drop it into
// libs/libigl_changes/include/igl/slim.cpp behind a #ifdef PASTIX6 branch.
//
// PaStiX 5.x had an upstream Eigen wrapper (Eigen/PaStiXSupport.h); PaStiX 6.x
// dropped C-API compatibility — completely new entry points (pastixInit /
// pastix_task_analyze / pastix_task_numfact / pastix_task_solve), spm-based
// matrix description, structured pastix_data_t. So we wrap by hand.
//
// The matrix MUST be column-major CSC, base-0, double-precision, SPD. SLIM's
// L = AᵀWA + λI fits all four. analyzePattern() captures the sparsity once;
// factorize() updates only values on subsequent iters.
#pragma once

#include <Eigen/SparseCore>

extern "C" {
#include <pastix.h>
#include <spm.h>
}

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace igl {

class Pastix6LLT {
public:
    using SparseMatrix = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
    using VectorXd = Eigen::VectorXd;

    Pastix6LLT() {
        pastixInitParam(iparm_, dparm_);
        // SPD with LLT factorisation; matches Eigen::CholmodSupernodalLLT.
        iparm_[IPARM_FACTORIZATION] = PastixFactPOTRF;
        // Quiet by default; bump to PastixVerboseNo / Yes for debugging.
        iparm_[IPARM_VERBOSE]       = PastixVerboseNot;
        // Threading: 0 = PaStiX auto via hwloc, which sizes to physical cores
        // (not logical / nproc / hardware_concurrency — that would oversubscribe
        // SMT). Users can override with the PASTIX_NUM_THREADS env var or by
        // calling setThreads(); PaStiX does not read OMP_NUM_THREADS.
        iparm_[IPARM_THREAD_NBR]    = 0;
        pastixInit(&data_, /*comm=*/0, iparm_, dparm_);
    }

    ~Pastix6LLT() {
        if (analyzed_) {
            spmExit(&spm_);
        }
        pastixFinalize(&data_);
    }

    Pastix6LLT(const Pastix6LLT&) = delete;
    Pastix6LLT& operator=(const Pastix6LLT&) = delete;

    void setThreads(int n) { iparm_[IPARM_THREAD_NBR] = n; }

    // analyze + numfact in one call, like Eigen's compute().
    Pastix6LLT& compute(const SparseMatrix& L) {
        if (!analyzed_) analyzePattern(L);
        factorize(L);
        return *this;
    }

    void analyzePattern(const SparseMatrix& L) {
        // Eigen stores symmetric matrices with both triangles populated; PaStiX
        // SpmSymmetric expects exactly one triangle. We extract the lower
        // triangle into a fresh CSC and feed that. spmCheckAndCorrect would
        // also symmetrize but it allocates internally and complicates the
        // value-update path used in factorize().
        if (analyzed_) {
            spmExit(&spm_);
        }
        spmInit(&spm_);
        spm_.mtxtype = SpmSymmetric;
        spm_.flttype = SpmDouble;
        spm_.fmttype = SpmCSC;
        spm_.baseval = 0;
        spm_.n       = static_cast<pastix_int_t>(L.rows());
        spm_.dof     = 1;
        spm_.layout  = SpmColMajor;

        const pastix_int_t n = static_cast<pastix_int_t>(L.rows());
        const int* outer = L.outerIndexPtr();
        const int* inner = L.innerIndexPtr();

        // Count lower-triangular nnz first (column j, rows i >= j).
        pastix_int_t lower_nnz = 0;
        for (pastix_int_t j = 0; j < n; ++j) {
            for (int p = outer[j]; p < outer[j + 1]; ++p) {
                if (inner[p] >= j) ++lower_nnz;
            }
        }

        spm_.nnz = lower_nnz;
        spmUpdateComputedFields(&spm_);

        spm_.colptr = static_cast<pastix_int_t*>(std::malloc((n + 1) * sizeof(pastix_int_t)));
        spm_.rowptr = static_cast<pastix_int_t*>(std::malloc(lower_nnz * sizeof(pastix_int_t)));
        spm_.values = std::malloc(lower_nnz * sizeof(double));

        // Cache the original-CSC index for each lower-triangular slot so
        // factorize() can refresh values in O(nnz_lower) without re-scanning.
        value_src_idx_.resize(lower_nnz);

        pastix_int_t out = 0;
        const pastix_int_t* col_start = spm_.colptr;
        (void)col_start;
        static_cast<pastix_int_t*>(spm_.colptr)[0] = 0;
        for (pastix_int_t j = 0; j < n; ++j) {
            for (int p = outer[j]; p < outer[j + 1]; ++p) {
                if (inner[p] >= j) {
                    static_cast<pastix_int_t*>(spm_.rowptr)[out] =
                        static_cast<pastix_int_t>(inner[p]);
                    value_src_idx_[out] = p;
                    ++out;
                }
            }
            static_cast<pastix_int_t*>(spm_.colptr)[j + 1] = out;
        }

        // Initial value copy.
        const double* src = L.valuePtr();
        double* dst = static_cast<double*>(spm_.values);
        for (pastix_int_t k = 0; k < lower_nnz; ++k) {
            dst[k] = src[value_src_idx_[k]];
        }

        pastix_task_analyze(data_, &spm_);
        analyzed_ = true;
        if (!logged_threads_) {
            std::cerr << "[pastix] IPARM_THREAD_NBR=" << iparm_[IPARM_THREAD_NBR]
                      << " (matrix n=" << n << ", lower_nnz=" << lower_nnz << ")"
                      << std::endl;
            logged_threads_ = true;
        }
    }

    void factorize(const SparseMatrix& L) {
        if (!analyzed_) {
            throw std::runtime_error("Pastix6LLT: factorize() before analyzePattern()");
        }
        // Sparsity is fixed across SLIM iters — just refresh the lower-tri
        // values using the index map captured in analyzePattern().
        const double* src = L.valuePtr();
        double* dst = static_cast<double*>(spm_.values);
        const auto n = value_src_idx_.size();
        for (std::size_t k = 0; k < n; ++k) {
            dst[k] = src[value_src_idx_[k]];
        }
        pastix_task_numfact(data_, &spm_);
    }

    // Not const: mutates pastix internal state (factorization context).
    VectorXd solve(const VectorXd& rhs) {
        const pastix_int_t n = spm_.n;
        if (rhs.size() % n != 0) {
            throw std::runtime_error(
                "Pastix6LLT::solve: rhs size " + std::to_string(rhs.size()) +
                " is not a multiple of system size " + std::to_string(n));
        }
        VectorXd x = rhs;
        const pastix_int_t nrhs = static_cast<pastix_int_t>(rhs.size() / n);
        // pastix_task_solve writes the answer in place over the rhs buffer.
        pastix_task_solve(data_, n, nrhs, x.data(), n);
        return x;
    }

private:
    pastix_data_t* data_ = nullptr;
    pastix_int_t   iparm_[IPARM_SIZE]{};
    double         dparm_[DPARM_SIZE]{};
    spmatrix_t     spm_{};
    std::vector<int> value_src_idx_;
    bool           analyzed_ = false;
    bool           logged_threads_ = false;
};

} // namespace igl
