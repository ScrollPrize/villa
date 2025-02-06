#pragma once

#include <Eigen/Sparse>
#include <vector>

// Include the standalone Pardiso header from pardiso-project.org
#include "pardiso.h"

/// A simple double-precision (double) wrapper around the standalone Pardiso solver.
/// This class handles row-major CSR conversion from Eigen's (column-major) sparse matrix,
/// and calls the Pardiso API with phase=11/22/33 for analysis/factorization/solve.
class PardisoSolver
{
public:
    PardisoSolver();
    ~PardisoSolver();

    /// Symbolic analysis (phase = 11). Convert matrix pattern once here.
    void analyzePattern(const Eigen::SparseMatrix<double> &A);

    /// Numeric factorization (phase = 22). Recompute factors if values change.
    void factorize(const Eigen::SparseMatrix<double> &A);

    /// Solve (phase = 33). Returns x for the system Ax=b.
    Eigen::VectorXd solve(const Eigen::VectorXd &rhs);

    /// Helper to release Pardiso memory. Called automatically by destructor.
    void releaseMemory();

    /// True if analyzePattern() was called
    bool isAnalyzed()   const { return analyzed_;   }
    /// True if factorize()    was called
    bool isFactorized() const { return factorized_; }

private:
    /// Convert from Eigen's default col-major format to row-major CSR
    void convertToCSR(const Eigen::SparseMatrix<double> &A);

private:
    // The internal Pardiso pointers (size 64).
    void* pt_[64];

    // Pardiso integer parameters
    int iparm_[64];
    // Pardiso double parameters
    double dparm_[64];

    // Matrix type: 2 => SPD, -2 => indefinite, 11 => unsymmetric, etc.
    int mtype_;

    // Basic Pardiso control
    int maxfct_, mnum_, phase_, error_, msglvl_;

    // Matrix dimension
    int n_;

    // CSR data (row-major)
    std::vector<int>    rowPtr_; // size n+1
    std::vector<int>    colIdx_; // size nnz
    std::vector<double> vals_;   // size nnz

    // IA, JA, Avals for Pardiso
    Eigen::VectorXi II, JJ;
    Eigen::VectorXd KK;

    // Bookkeeping
    bool analyzed_   = false;
    bool factorized_ = false;
};
