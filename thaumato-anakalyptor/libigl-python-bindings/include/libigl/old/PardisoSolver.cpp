#include "PardisoSolver.h"

#include <cstring>   // for memset
#include <iostream>  // for std::cerr

PardisoSolver::PardisoSolver()
{
    // Clear internal arrays
    std::memset(pt_,    0, sizeof(pt_));
    std::memset(iparm_, 0, sizeof(iparm_));
    std::memset(dparm_, 0, sizeof(dparm_));

    // By default, let's use a real SPD system (mtype=2).
    // Change to -2 for indefinite, 11 for unsymmetric, etc. as needed.
    mtype_ = 2;

    // Some recommended Pardiso default parameters
    iparm_[0]  = 1;   // Do not use Pardiso defaults
    iparm_[1]  = 2;   // Metis reordering
    iparm_[2]  = 0;   // #threads: 0 => use all cores (or environment var OMP_NUM_THREADS)
    iparm_[3]  = 0;   // For iterative-direct methods (not used here)
    iparm_[4]  = 0;   // No user fill-in permutation
    iparm_[5]  = 0;   // Write solution into x? (0 => keep separate)
    iparm_[6]  = 0;   // Not in-place
    iparm_[7]  = 1;   // Max number of iterative refinement steps
    iparm_[8]  = 0;   // Reserved
    iparm_[9]  = 8;  // Pivot perturbation
    iparm_[10] = 0;   // Use nonsymmetric permutation scaling MPS
    iparm_[11] = 0;   // Solve Ax=b
    iparm_[12] = 0;   // Not storing solver data
    iparm_[13] = 1;   // Output, 0 => no
    iparm_[14] = 0;   // Pivoting mode
    iparm_[15] = 0;   // Perturbation
    iparm_[16] = 0;   
    iparm_[17] = 1;  // Print level: -1 => none
    iparm_[18] = 1;  // Another print level
    iparm_[19] = 0;   

    // Basic controlling
    maxfct_ = 1;  // Max # of factorizations
    mnum_   = 1;  // Which factorization to use
    msglvl_ = 0;  // 0 => no printing
    error_  = 0;
}

PardisoSolver::~PardisoSolver()
{
    releaseMemory();
}

void PardisoSolver::releaseMemory()
{
    // If we have done analysis or factorization, then free Pardiso structures
    if (analyzed_ || factorized_)
    {
        phase_ = -1; // -1 => release all internal memory
        int nrhs = 0;
        pardiso(
            pt_,            // Internal solver address pointer
            &maxfct_,       // max # of factorizations
            &mnum_,         // which factorization
            &mtype_,        // matrix type
            &phase_,        // phase = -1
            &n_,            // dimension
            nullptr,        // a = null
            nullptr,        // ia
            nullptr,        // ja
            nullptr,        // perm
            &nrhs,          // nrhs=0
            iparm_,         // iparm
            &msglvl_,       // msglvl
            nullptr,        // b
            nullptr,        // x
            &error_,        // error
            dparm_          // dparm
        );

        analyzed_   = false;
        factorized_ = false;
    }
}

void PardisoSolver::analyzePattern(const Eigen::SparseMatrix<double> &A)
{
    // Convert from column-major (default in Eigen) to row-major CSR for Pardiso
    convertToCSR(A);
    std::cout << "Converted to CSR\n";
    n_ = A.rows();

    // phase=11 => symbolic factorization
    phase_ = 11;
    int nrhs = 0;

    std::cout << "Calling Pardiso\n";

    pardiso(
        pt_,            // pt
        &maxfct_,       // maxfact
        &mnum_,         // mnum
        &mtype_,        // mtype
        &phase_,        // phase=11
        &n_,            // dimension
        vals_.data(),   // a
        rowPtr_.data(), // ia
        colIdx_.data(), // ja
        nullptr,        // perm
        &nrhs,          // nrhs=0
        iparm_,         // iparm
        &msglvl_,       // msglvl
        nullptr,        // b => null
        nullptr,        // x => null
        &error_,        // error
        dparm_          // dparm
    );

    std::cout << "Pardiso called\n";

    if (error_ != 0)
    {
        std::cerr << "[PardisoSolver] analyzePattern() error code = " << error_ << std::endl;
    }

    analyzed_   = true;
    factorized_ = false;
}

void PardisoSolver::factorize(const Eigen::SparseMatrix<double> &A)
{
    // If we have not yet done the symbolic analysis, do so now
    if (!analyzed_) {
        analyzePattern(A);
    }
    else
    {
        // Possibly check if pattern changed, if so call releaseMemory() + analyzePattern(A).
        // Otherwise, we can just update the numeric values.
        convertToCSR(A);
    }

    // phase=22 => numeric factorization
    phase_ = 22;
    int nrhs = 0;

    pardiso(
        pt_,            // pt
        &maxfct_,       // maxfact
        &mnum_,         // mnum
        &mtype_,        // mtype
        &phase_,        // phase=22
        &n_,            // dimension
        vals_.data(),   // a
        rowPtr_.data(), // ia
        colIdx_.data(), // ja
        nullptr,        // perm
        &nrhs,          // nrhs=0
        iparm_,         // iparm
        &msglvl_,       // msglvl
        nullptr,        // b => null
        nullptr,        // x => null
        &error_,        // error
        dparm_          // dparm
    );

    if (error_ != 0)
    {
        std::cerr << "[PardisoSolver] factorize() error code = " << error_ << std::endl;
    }

    factorized_ = true;
}

Eigen::VectorXd PardisoSolver::solve(const Eigen::VectorXd &rhs)
{
    if (!factorized_)
    {
        std::cerr << "[PardisoSolver] solve() called but factorize() not done yet!\n";
        return Eigen::VectorXd::Zero(n_);
    }

    phase_ = 33;  // solve & iterative refinement
    int nrhs = 1; // We have exactly 1 RHS

    // Copy rhs into a standard double array
    std::vector<double> b(rhs.data(), rhs.data() + rhs.size());
    // Prepare storage for solution
    std::vector<double> x(n_, 0.0);

    pardiso(
        pt_,            // pt
        &maxfct_,       // maxfact
        &mnum_,         // mnum
        &mtype_,        // mtype
        &phase_,        // phase=33
        &n_,            // dimension
        vals_.data(),   // a
        rowPtr_.data(), // ia
        colIdx_.data(), // ja
        nullptr,        // perm
        &nrhs,          // &nrhs=1
        iparm_,         // iparm
        &msglvl_,       // msglvl
        b.data(),       // b array
        x.data(),       // x array
        &error_,        // error
        dparm_          // dparm
    );

    if (error_ != 0)
    {
        std::cerr << "[PardisoSolver] solve() error code = " << error_ << std::endl;
        return Eigen::VectorXd::Zero(n_);
    }

    // Wrap the solution back into an Eigen vector
    return Eigen::Map<Eigen::VectorXd>(x.data(), x.size());
}

void PardisoSolver::convertToCSR(const Eigen::SparseMatrix<double> &A)
{
    // Number of nonzeros
    int nnz = A.nonZeros();

    // Prepare storage for CRS format
    std::vector<int> row_ptr(A.rows() + 1, 0);
    std::vector<int> col_idx;
    std::vector<double> values;
    col_idx.reserve(nnz);
    values.reserve(nnz);

    // Convert Eigen sparse matrix to CRS format
    for (int k = 0; k < A.outerSize(); ++k) {
        row_ptr[k] = col_idx.size();  // Start index of this row in JA and Avals
        for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
            if (it.row() <= it.col()) {  // Store only upper triangular part (if symmetric)
                col_idx.push_back(it.col());
                values.push_back(it.value());
            }
        }
    }
    row_ptr[A.rows()] = col_idx.size();  // End index of last row

    // Convert to Eigen types
    II = Eigen::Map<Eigen::VectorXi>(row_ptr.data(), row_ptr.size());
    JJ = Eigen::Map<Eigen::VectorXi>(col_idx.data(), col_idx.size());
    KK = Eigen::Map<Eigen::VectorXd>(values.data(), values.size());
}
