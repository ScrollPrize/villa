/* This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. See https://mozilla.org/MPL/2.0/. */
#ifndef IGL_SLIM_ASSEMBLY_H
#define IGL_SLIM_ASSEMBLY_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
#ifdef PASTIX6
#include "pastix6_solver.h"
#endif

namespace igl::slim
{
namespace detail
{
constexpr int triangle_corners = 3;
constexpr int uv_components = 2;
constexpr int triangle_dofs = triangle_corners * uv_components;
constexpr int triangle_entries = triangle_dofs * triangle_dofs;

constexpr int matrix_slot(int row_axis, int col_axis, int row_corner, int col_corner)
{
    return ((row_axis * uv_components + col_axis) * triangle_corners + row_corner) * triangle_corners + col_corner;
}

inline void check_triangle_dimensions(Eigen::Index vertices, Eigen::Index faces)
{
    if (vertices <= 0 || faces <= 0)
        throw std::invalid_argument("SLIM requires a nonempty triangle mesh");
    constexpr auto limit = std::numeric_limits<int>::max();
    // Sparse storage indices also bound the temporary triplet matrix.
    if (vertices > limit / uv_components || faces > limit / triangle_entries)
        throw std::length_error("SLIM triangle assembly exceeds 32-bit sparse storage");
}

inline void validate_triangle_input(Eigen::Index vertices, const Eigen::MatrixXi& faces)
{
    check_triangle_dimensions(vertices, faces.rows());
    std::vector<unsigned char> referenced(static_cast<std::size_t>(vertices), 0);
    for (Eigen::Index f = 0; f < faces.rows(); ++f)
        for (int corner = 0; corner < triangle_corners; ++corner) {
            const int vertex = faces(f, corner);
            if (vertex < 0 || vertex >= vertices)
                throw std::invalid_argument("SLIM face vertex is outside the vertex array");
            referenced[vertex] = 1;
        }
    // Both quadratic routes require every proximal diagonal to exist.
    if (std::find(referenced.begin(), referenced.end(), 0) != referenced.end())
        throw std::invalid_argument("SLIM requires every vertex to belong to a face; remove unreferenced vertices");
}

/* Per-triangle assembly of the 2D quadratic, using the native coordinate
   ordering and mass convention. The topology is fixed between precomputes. */
struct TriangleAssembly {
    Eigen::SparseMatrix<double> matrix;
    Eigen::VectorXd rhs;
    Eigen::Matrix<double, Eigen::Dynamic, triangle_dofs> gradients;
    std::vector<std::array<int, triangle_entries>> matrix_offsets;

    template <class Data>
    explicit TriangleAssembly(const Data& s)
    {
        check_triangle_dimensions(s.v_n, s.f_n);
        gradients.resize(s.f_n, triangle_dofs);
        matrix_offsets.resize(s.f_n);
        matrix.resize(uv_components * s.v_n, uv_components * s.v_n);
        {
            std::vector<Eigen::Triplet<double>> entries;
            entries.reserve(static_cast<std::size_t>(s.f_n) * triangle_entries);
            for (int face = 0; face < s.f_n; ++face) {
                for (int corner = 0; corner < triangle_corners; ++corner) {
                    gradients(face, corner) = s.Dx.coeff(face, s.F(face, corner));
                    gradients(face, corner + triangle_corners) = s.Dy.coeff(face, s.F(face, corner));
                }
                for (int row_axis = 0; row_axis < uv_components; ++row_axis)
                    for (int col_axis = 0; col_axis < uv_components; ++col_axis)
                        for (int row_corner = 0; row_corner < triangle_corners; ++row_corner)
                            for (int col_corner = 0; col_corner < triangle_corners; ++col_corner)
                                entries.emplace_back(s.F(face, row_corner) + row_axis * s.v_n, s.F(face, col_corner) + col_axis * s.v_n, 1.);
            }
            matrix.setFromTriplets(entries.begin(), entries.end());
            matrix.makeCompressed();
        }
        for (int face = 0; face < s.f_n; ++face) {
            for (int row_axis = 0; row_axis < uv_components; ++row_axis)
                for (int col_axis = 0; col_axis < uv_components; ++col_axis)
                    for (int row_corner = 0; row_corner < triangle_corners; ++row_corner)
                        for (int col_corner = 0; col_corner < triangle_corners; ++col_corner) {
                            const int row = s.F(face, row_corner) + row_axis * s.v_n;
                            const int col = s.F(face, col_corner) + col_axis * s.v_n;
                            const int begin = matrix.outerIndexPtr()[col], end = matrix.outerIndexPtr()[col + 1];
                            const int* p = std::lower_bound(matrix.innerIndexPtr() + begin, matrix.innerIndexPtr() + end, row);
                            if (p == matrix.innerIndexPtr() + end || *p != row)
                                throw std::runtime_error("Missing SLIM triangle assembly entry");
                            matrix_offsets[face][matrix_slot(row_axis, col_axis, row_corner, col_corner)] =
                                static_cast<int>(p - matrix.innerIndexPtr());
                        }
        }
    }

    template <class Data>
    void update(const Data& s)
    {
        std::fill(matrix.valuePtr(), matrix.valuePtr() + matrix.nonZeros(), 0.);
        rhs.setZero(uv_components * s.v_n);
        Eigen::VectorXd rhs_odd = Eigen::VectorXd::Zero(uv_components * s.v_n);
        Eigen::VectorXi occurrences = Eigen::VectorXi::Zero(s.v_n);
        /* The weighted rows are W row 0/Dx, W row 0/Dy, W row 1/Dx,
           W row 1/Dy. Visit these blocks, then faces, in sparse-product order. */
        for (int block = 0; block < uv_components * uv_components; ++block)
            for (int face = 0; face < s.f_n; ++face) {
                const int weight_offset = uv_components * (block / uv_components);
                const int gradient_offset = triangle_corners * (block % uv_components);
                const int rotation_offset = uv_components * (block % uv_components);
                double weighted_gradient[uv_components][triangle_corners];
                for (int axis = 0; axis < uv_components; ++axis)
                    for (int corner = 0; corner < triangle_corners; ++corner)
                        weighted_gradient[axis][corner] = gradients(face, gradient_offset + corner) * s.W(face, weight_offset + axis);
                const double weighted_rotation =
                    (s.W(face, weight_offset) * s.Ri(face, rotation_offset) + s.W(face, weight_offset + 1) * s.Ri(face, rotation_offset + 1)) *
                    s.M(face);
                for (int row_axis = 0; row_axis < uv_components; ++row_axis)
                    for (int col_axis = 0; col_axis < uv_components; ++col_axis)
                        for (int row_corner = 0; row_corner < triangle_corners; ++row_corner)
                            for (int col_corner = 0; col_corner < triangle_corners; ++col_corner) {
                                const int slot = matrix_slot(row_axis, col_axis, row_corner, col_corner);
                                matrix.valuePtr()[matrix_offsets[face][slot]] +=
                                    weighted_gradient[row_axis][row_corner] * s.M(face) * weighted_gradient[col_axis][col_corner];
                            }
                // Eigen's row-major sparse product alternates two accumulators.
                for (int corner = 0; corner < triangle_corners; ++corner) {
                    const int vertex = s.F(face, corner);
                    const bool odd = (occurrences(vertex)++ % 2) != 0;
                    for (int axis = 0; axis < uv_components; ++axis) {
                        const int index = vertex + axis * s.v_n;
                        if (odd)
                            rhs_odd(index) += weighted_gradient[axis][corner] * weighted_rotation;
                        else
                            rhs(index) += weighted_gradient[axis][corner] * weighted_rotation;
                    }
                }
            }
        rhs += rhs_odd;
        rhs += s.proximal_p * Eigen::Map<const Eigen::VectorXd>(s.V_o.data(), uv_components * s.v_n);
        matrix.diagonal().array() += s.proximal_p;
    }
};
}  // namespace detail

/* Copies retain the SLIM problem but rebuild mutable numerical caches.
   Moves transfer ownership. Destruction releases both assembly and solver. */
struct Cache {
    std::unique_ptr<detail::TriangleAssembly> triangles;
#ifdef PASTIX6
    std::unique_ptr<igl::Pastix6LLT> solver;
#endif
    Cache() = default;
    Cache(const Cache&) {}
    Cache& operator=(const Cache& other)
    {
        if (this != &other)
            *this = Cache{};
        return *this;
    }
    Cache(Cache&&) = default;
    Cache& operator=(Cache&&) = default;
};
}  // namespace igl::slim
#endif
