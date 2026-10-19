#include <igl/slim.h>
#include <igl/readOBJ.h>
#include "vc_test.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace
{
double matrix_error = 0., rhs_error = 0., lifecycle_error = 0.;
int systems = 0;
int input_checks = 0;
int comparison_rejections = 0;

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void same(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b)
{
    require(a.rows() == b.rows() && a.cols() == b.cols(), "Lifecycle shape mismatch");
    require(a.allFinite() && b.allFinite(), "Nonfinite lifecycle result");
    const double error = (a - b).cwiseAbs().maxCoeff();
    lifecycle_error = std::max(lifecycle_error, error);
    require(error < 1e-9, "Lifecycle changed solution");
}

void compare_systems(const Eigen::SparseMatrix<double>& direct, const Eigen::VectorXd& rhs, const Eigen::SparseMatrix<double>& reference, const Eigen::VectorXd& reference_rhs)
{
    require(
        direct.rows() == reference.rows() && direct.cols() == reference.cols() && rhs.size() == reference_rhs.size() &&
            rhs.size() == direct.rows() && rhs.size() > 0,
        "System shape mismatch");
    require(rhs.allFinite() && reference_rhs.allFinite(), "Nonfinite system RHS");
    for (const auto* matrix : {&direct, &reference})
        for (int col = 0; col < matrix->outerSize(); ++col)
            for (Eigen::SparseMatrix<double>::InnerIterator entry(*matrix, col); entry; ++entry)
                require(std::isfinite(entry.value()), "Nonfinite matrix coefficient");
    Eigen::SparseMatrix<double> delta = direct - reference;
    for (int col = 0; col < delta.outerSize(); ++col)
        for (Eigen::SparseMatrix<double>::InnerIterator entry(delta, col); entry; ++entry) {
            const double error = std::abs(entry.value()) / (1e-10 + 1e-10 * std::abs(reference.coeff(entry.row(), entry.col())));
            require(std::isfinite(error), "Nonfinite matrix comparison error");
            if (error > 1.) {
                std::ostringstream message;
                message << std::setprecision(17) << "Matrix mismatch at (" << entry.row() << ", " << entry.col()
                        << "): actual=" << direct.coeff(entry.row(), entry.col())
                        << ", expected=" << reference.coeff(entry.row(), entry.col()) << ", scaled error=" << error;
                throw std::runtime_error(message.str());
            }
            matrix_error = std::max(matrix_error, error);
        }
    const Eigen::ArrayXd rhs_errors = (rhs - reference_rhs).array().abs() / (1e-10 + 1e-10 * reference_rhs.array().abs());
    require(rhs_errors.allFinite(), "Nonfinite RHS comparison error");
    Eigen::Index worst_row;
    const double maximum_rhs_error = rhs_errors.maxCoeff(&worst_row);
    if (maximum_rhs_error > 1.) {
        std::ostringstream message;
        message << std::setprecision(17) << "RHS mismatch at row " << worst_row << ": actual=" << rhs(worst_row)
                << ", expected=" << reference_rhs(worst_row) << ", scaled error=" << maximum_rhs_error;
        throw std::runtime_error(message.str());
    }
    rhs_error = std::max(rhs_error, maximum_rhs_error);
    require(direct.nonZeros() == reference.nonZeros(), "Matrix sparsity changed");
}

void check_system(igl::SLIMData& s, int iteration = 0)
{
    igl::slim::update_weights_and_closest_rotations(s, s.V_o);
    Eigen::SparseMatrix<double> direct, reference;
    igl::slim::build_linear_system(s, direct);
    const Eigen::VectorXd rhs = s.rhs;
    igl::slim::build_linear_system_generic(s, reference);
    try {
        compare_systems(direct, rhs, reference, s.rhs);
    } catch (const std::runtime_error& error) {
        throw std::runtime_error("iteration " + std::to_string(iteration) + ": " + error.what());
    }
    ++systems;
}

void initialize(igl::SLIMData& s, bool alternate, igl::MappingEnergyType energy)
{
    Eigen::MatrixXd v(4, 3), uv(4, 2), bc(2, 2);
    Eigen::MatrixXi f(2, 3);
    Eigen::VectorXi b(2);
    v << 0, 0, 0, 1.1, 0, .04, 1, 1, .08, 0, 1.2, -.03;
    uv << 0, 0, 1.04, .03, 1.05, 1.1, -.04, 1.15;
    if (alternate)
        f << 0, 1, 3, 1, 2, 3;
    else
        f << 0, 1, 2, 0, 2, 3;
    b << 0, 2;
    bc.row(0) = uv.row(0);
    bc.row(1) = uv.row(2);
    bc.array() += .025;
    igl::slim_precompute(v, f, uv, s, energy, b, bc, 3.7);
}

void tetrahedron(igl::SLIMData& s)
{
    Eigen::MatrixXd v(4, 3), uv, bc(0, 3);
    Eigen::MatrixXi f(1, 4);
    Eigen::VectorXi b(0);
    v << 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1;
    f << 0, 1, 2, 3;
    uv = v;
    uv.col(0) *= 1.05;
    uv.col(1) *= .95;
    igl::slim_precompute(v, f, uv, s, igl::MappingEnergyType::SYMMETRIC_DIRICHLET, b, bc, 0.);
    require(s.dim == 3 && !s.cache.triangles && !s.cache.solver, "Stale cache on 3D reinitialization");
}

template <class Exception, class Function>
void rejects(Function function, const char* diagnostic, int& counter = input_checks)
{
    bool rejected = false;
    try {
        function();
    } catch (const Exception& error) {
        require(std::string(error.what()).find(diagnostic) != std::string::npos, "Rejection diagnostic did not identify the failure");
        rejected = true;
    }
    require(rejected, "Expected rejection did not occur");
    ++counter;
}

TEST_CASE("SLIM comparisons reject nonfinite matrices and right-hand sides")
{
    for (double invalid : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()})
        for (int operand = 0; operand < 4; ++operand) {
            Eigen::SparseMatrix<double> direct(2, 2), reference(2, 2);
            direct.setIdentity();
            reference.setIdentity();
            Eigen::VectorXd rhs = Eigen::VectorXd::Ones(2), reference_rhs = rhs;
            if (operand == 0)
                direct.coeffRef(0, 0) = invalid;
            else if (operand == 1)
                reference.coeffRef(0, 0) = invalid;
            else if (operand == 2)
                rhs(0) = invalid;
            else
                reference_rhs(0) = invalid;
            rejects<std::runtime_error>([&] { compare_systems(direct, rhs, reference, reference_rhs); }, "Nonfinite", comparison_rejections);
        }

    igl::SLIMData data;
    initialize(data, false, igl::MappingEnergyType::SYMMETRIC_DIRICHLET);
    data.proximal_p = std::numeric_limits<double>::quiet_NaN();
    rejects<std::runtime_error>([&] { check_system(data); }, "Nonfinite", comparison_rejections);
}

TEST_CASE("SLIM comparisons identify finite coefficient and RHS mismatches")
{
    Eigen::SparseMatrix<double> reference(2, 2);
    reference.setIdentity();
    Eigen::SparseMatrix<double> direct = reference;
    const Eigen::VectorXd reference_rhs = Eigen::VectorXd::Ones(2);
    Eigen::VectorXd rhs = reference_rhs;
    direct.coeffRef(1, 1) += 1.;
    rejects<std::runtime_error>([&] { compare_systems(direct, rhs, reference, reference_rhs); }, "Matrix mismatch at (1, 1)", comparison_rejections);
    direct = reference;
    rhs(1) += 1.;
    rejects<std::runtime_error>([&] { compare_systems(direct, rhs, reference, reference_rhs); }, "RHS mismatch at row 1", comparison_rejections);
}

TEST_CASE("SLIM rejects invalid input without invalidating a usable problem")
{
    using igl::slim::detail::check_triangle_dimensions;
    constexpr Eigen::Index limit = std::numeric_limits<int>::max();
    check_triangle_dimensions(limit / 2, limit / 36);
    rejects<std::length_error>([&] { check_triangle_dimensions(limit / 2 + 1, 1); }, "32-bit");
    rejects<std::length_error>([&] { check_triangle_dimensions(3, limit / 36 + 1); }, "32-bit");
    rejects<std::invalid_argument>([] { check_triangle_dimensions(0, 1); }, "nonempty");
    rejects<std::invalid_argument>([] { check_triangle_dimensions(3, 0); }, "nonempty");

    igl::SLIMData data;
    initialize(data, false, igl::MappingEnergyType::SYMMETRIC_DIRICHLET);
    igl::slim_solve(data, 1);
    const auto original = data.V_o;
    const auto* solver = data.cache.solver.get();
    Eigen::MatrixXd vertices = data.V, uv = data.V_o;
    vertices.conservativeResize(5, 3);
    vertices.row(4) << 2, 2, 0;
    uv.conservativeResize(5, 2);
    uv.row(4) << 2, 2;
    const auto faces = data.F;
    const auto constraints = data.b;
    const auto targets = data.bc;
    rejects<std::invalid_argument>(
        [&] { igl::slim_precompute(vertices, faces, uv, data, igl::MappingEnergyType::SYMMETRIC_DIRICHLET, constraints, targets, 3.7); },
        "unreferenced");
    require(data.cache.solver.get() == solver, "Rejected input invalidated a usable problem");
    same(original, data.V_o);
    for (const int invalid : {-1, 4}) {
        Eigen::MatrixXi malformed = faces;
        malformed(0, 1) = invalid;
        rejects<std::invalid_argument>(
            [&] {
                igl::slim_precompute(data.V, malformed, data.V_o, data, igl::MappingEnergyType::SYMMETRIC_DIRICHLET, constraints, targets, 3.7);
            },
            "outside");
    }
    igl::slim_solve(data, 1);
    require(data.V_o.allFinite(), "A rejected input damaged subsequent solving");
}

template <class Function>
void for_each_energy(Function check)
{
    const std::pair<igl::MappingEnergyType, const char*> energies[] =
        {{igl::MappingEnergyType::ARAP, "ARAP"},
         {igl::MappingEnergyType::LOG_ARAP, "LOG_ARAP"},
         {igl::MappingEnergyType::SYMMETRIC_DIRICHLET, "SYMMETRIC_DIRICHLET"},
         {igl::MappingEnergyType::CONFORMAL, "CONFORMAL"},
         {igl::MappingEnergyType::EXP_CONFORMAL, "EXP_CONFORMAL"},
         {igl::MappingEnergyType::EXP_SYMMETRIC_DIRICHLET, "EXP_SYMMETRIC_DIRICHLET"}};
    for (const auto& [energy, name] : energies) {
        try {
            check(energy);
        } catch (const std::exception& error) {
            throw std::runtime_error(std::string(name) + ": " + error.what());
        }
    }
}

TEST_CASE("SLIM triangle assembly matches generic assembly for each energy")
{
    for_each_energy([](auto energy) {
        igl::SLIMData s;
        initialize(s, false, energy);
        for (int iteration = 0; iteration < 3; ++iteration) {
            check_system(s, iteration);
            igl::slim_solve(s, 1);
        }
    });
}

TEST_CASE("SLIM copies rebuild caches and moves transfer ownership")
{
    static_assert(std::is_copy_constructible_v<igl::SLIMData>);
    static_assert(std::is_copy_assignable_v<igl::SLIMData>);
    static_assert(std::is_move_constructible_v<igl::SLIMData>);
    for_each_energy([](auto energy) {
        igl::SLIMData s;
        initialize(s, false, energy);
        igl::slim_solve(s, 3);
        igl::SLIMData copied = s, assigned;
        initialize(assigned, true, energy);
        igl::slim_solve(assigned, 1);
        assigned = s;
        require(
            !copied.cache.triangles && !copied.cache.solver && !assigned.cache.triangles && !assigned.cache.solver,
            "Copy retained mutable solver state");
        igl::slim_solve(s, 1);
        igl::slim_solve(copied, 1);
        igl::slim_solve(assigned, 1);
        same(s.V_o, copied.V_o);
        same(s.V_o, assigned.V_o);
        auto* solver = copied.cache.solver.get();
        igl::SLIMData moved = std::move(copied);
        require(moved.cache.solver.get() == solver && !copied.cache.solver, "Move lost solver ownership");
        igl::slim_solve(s, 1);
        igl::slim_solve(moved, 1);
        same(s.V_o, moved.V_o);
        assigned = std::move(moved);
        require(assigned.cache.solver.get() == solver && !moved.cache.solver, "Move assignment lost solver ownership");
        igl::slim_solve(s, 1);
        igl::slim_solve(assigned, 1);
        same(s.V_o, assigned.V_o);
    });
}

TEST_CASE("SLIM precomputation invalidates equal-size topology caches")
{
    for_each_energy([](auto energy) {
        igl::SLIMData s;
        initialize(s, false, energy);
        igl::slim_solve(s, 1);
        const auto original = s.cache.triangles->matrix;
        initialize(s, true, energy);
        require(!s.cache.solver && !s.cache.triangles && s.A.rows() == 0, "Recompute retained cache");
        check_system(s);
        const auto& changed = s.cache.triangles->matrix;
        require(original.nonZeros() == changed.nonZeros(), "Topology fixture needs equal nonzero counts");
        require(
            !std::equal(original.innerIndexPtr(), original.innerIndexPtr() + original.nonZeros(), changed.innerIndexPtr()),
            "Topology fixture needs different sparsity");
        igl::SLIMData fresh;
        initialize(fresh, true, energy);
        igl::slim_solve(s, 3);
        igl::slim_solve(fresh, 3);
        same(s.V_o, fresh.V_o);
    });
}

TEST_CASE("SLIM reinitialization supports triangle and tetrahedral assembly")
{
    for_each_energy([](auto energy) {
        igl::SLIMData s, fresh;
        initialize(s, false, energy);
        igl::slim_solve(s, 1);
        tetrahedron(s);
        check_system(s);
        igl::slim_solve(s, 2);
        require(s.V_o.allFinite() && !s.cache.triangles, "3D fallback failed");
        initialize(s, false, energy);
        check_system(s);
        igl::slim_solve(s, 2);
        initialize(fresh, false, energy);
        igl::slim_solve(fresh, 2);
        same(s.V_o, fresh.V_o);
    });
}

TEST_CASE("SLIM releases solver state when a problem is destroyed")
{
    for (int cycle = 0; cycle < 50; ++cycle) {
        igl::SLIMData s;
        initialize(s, cycle % 2, igl::MappingEnergyType::SYMMETRIC_DIRICHLET);
        igl::slim_solve(s, 2);
        require(s.V_o.allFinite(), "Destroyed-instance reuse failed");
    }
}

TEST_CASE("SLIM applies soft constraints to the requested coordinates")
{
    igl::SLIMData s;
    initialize(s, false, igl::MappingEnergyType::SYMMETRIC_DIRICHLET);
    igl::slim::update_weights_and_closest_rotations(s, s.V_o);
    s.soft_const_p = 0.;
    Eigen::SparseMatrix<double> expected, actual;
    igl::slim::build_linear_system(s, expected);
    Eigen::VectorXd expected_rhs = s.rhs;
    s.soft_const_p = 3.7;
    igl::slim::build_linear_system(s, actual);

    // The fixture constrains vertices 0 and 2, with all U coordinates before V.
    Eigen::VectorXd diagonal(8), rhs_increment(8);
    diagonal << 3.7, 0., 3.7, 0., 3.7, 0., 3.7, 0.;
    rhs_increment << 3.7 * s.bc(0, 0), 0., 3.7 * s.bc(1, 0), 0., 3.7 * s.bc(0, 1), 0., 3.7 * s.bc(1, 1), 0.;
    expected.diagonal() += diagonal;
    expected_rhs += rhs_increment;
    compare_systems(actual, s.rhs, expected, expected_rhs);
}

Eigen::MatrixXd vertex_uvs(Eigen::Index vertex_count, const Eigen::MatrixXi& faces, const Eigen::MatrixXd& texcoords, const Eigen::MatrixXi& face_texcoords)
{
    require(faces.cols() == 3, "Real OBJ requires triangular faces");
    require(
        texcoords.cols() >= 2 && texcoords.rows() > 0 && face_texcoords.rows() == faces.rows() && face_texcoords.cols() == 3,
        "Real OBJ requires UV coordinates for every face corner");
    igl::slim::detail::validate_triangle_input(vertex_count, faces);
    Eigen::MatrixXd uv = Eigen::MatrixXd::Constant(vertex_count, 2, std::numeric_limits<double>::quiet_NaN());
    for (int face = 0; face < faces.rows(); ++face)
        for (int corner = 0; corner < 3; ++corner) {
            const int vertex = faces(face, corner);
            const int texcoord = face_texcoords(face, corner);
            require(texcoord >= 0 && texcoord < texcoords.rows(), "Real OBJ UV index is outside the texture coordinate array");
            const Eigen::RowVector2d corner_uv = texcoords.row(texcoord).head<2>();
            require(corner_uv.allFinite(), "Real OBJ has nonfinite UV coordinates");
            if (uv.row(vertex).allFinite())
                require((uv.row(vertex).array() == corner_uv.array()).all(), "Real OBJ has conflicting UVs for one vertex");
            uv.row(vertex) = corner_uv;
        }
    return uv;
}

TEST_CASE("SLIM OBJ comparison maps face-corner UVs to vertices")
{
    Eigen::MatrixXi faces(2, 3), face_texcoords(2, 3);
    Eigen::MatrixXd texcoords(5, 2), expected(4, 2);
    faces << 0, 1, 2, 0, 2, 3;
    face_texcoords << 2, 3, 0, 4, 0, 1;
    texcoords << 1, 1, 0, 1, 0, 0, 1, 0, 0, 0;
    expected << 0, 0, 1, 0, 1, 1, 0, 1;
    require((vertex_uvs(4, faces, texcoords, face_texcoords).array() == expected.array()).all(), "OBJ vertex UV mapping differs");
}

TEST_CASE("SLIM OBJ comparison rejects missing, invalid and conflicting UVs")
{
    Eigen::MatrixXi faces(2, 3);
    faces << 0, 1, 2, 0, 2, 3;
    Eigen::MatrixXd texcoords(4, 2);
    texcoords << 0, 0, 1, 0, 1, 1, 0, 1;
    Eigen::MatrixXi quad(1, 4);
    quad << 0, 1, 2, 3;
    rejects<std::runtime_error>([&] { vertex_uvs(4, quad, texcoords, quad); }, "triangular faces");
    rejects<std::runtime_error>([&] { vertex_uvs(4, faces, Eigen::MatrixXd(), Eigen::MatrixXi()); }, "requires UV");
    rejects<std::runtime_error>([&] { vertex_uvs(4, faces, texcoords, Eigen::MatrixXi()); }, "requires UV");
    for (int index : {-1, 4}) {
        Eigen::MatrixXi invalid = faces;
        invalid(0, 0) = index;
        rejects<std::runtime_error>([&] { vertex_uvs(4, faces, texcoords, invalid); }, "UV index");
    }
    Eigen::MatrixXi conflicting = faces;
    conflicting(1, 0) = 3;
    rejects<std::runtime_error>([&] { vertex_uvs(4, faces, texcoords, conflicting); }, "conflicting UVs");
    texcoords(0, 0) = std::numeric_limits<double>::quiet_NaN();
    rejects<std::runtime_error>([&] { vertex_uvs(4, faces, texcoords, faces); }, "nonfinite UV");
}

std::pair<Eigen::Index, Eigen::Index> check_real_surface(const char* path)
{
    Eigen::MatrixXd v, tc, n;
    Eigen::MatrixXi f, ft, fn;
    require(igl::readOBJ(path, v, tc, n, f, ft, fn), "Real OBJ read failed");
    const Eigen::MatrixXd uv = vertex_uvs(v.rows(), f, tc, ft);
    igl::SLIMData s;
    Eigen::VectorXi b(0);
    Eigen::MatrixXd bc(0, 2);
    igl::slim_precompute(v, f, uv, s, igl::MappingEnergyType::SYMMETRIC_DIRICHLET, b, bc, 0.);
    for (int step = 0; step < 3; ++step) {
        if (step == 2) {
            s.b.resize(2);
            s.b << 0, v.rows() - 1;
            s.bc.resize(2, 2);
            s.bc.row(0) = s.V_o.row(0);
            s.bc.row(1) = s.V_o.row(v.rows() - 1);
            s.bc.array() += .125;
            s.soft_const_p = 3.7;
        }
        check_system(s, step);
        igl::slim_solve(s, 1);
    }
    return {v.rows(), f.rows()};
}

void write_summary(std::ostream& result, Eigen::Index vertices, Eigen::Index faces)
{
    result << std::setprecision(17) << "{\"systems\":" << systems << ",\"input_rejections\":" << input_checks
           << ",\"comparison_rejections\":" << comparison_rejections << ",\"matrix_max_scaled_error\":" << matrix_error
           << ",\"rhs_max_scaled_error\":" << rhs_error << ",\"lifecycle_max_absolute_error\":" << lifecycle_error
           << ",\"real_vertices\":" << vertices << ",\"real_faces\":" << faces << "}\n";
    result.flush();
    require(result.good(), "Cannot write regression result");
}
}  // namespace

int main(int argc, char** argv)
{
    try {
        require(argc == 1 || argc == 3, "Expected no arguments, or a real OBJ and unused result path");
        if (argc == 3)
            require(!std::ifstream(argv[2]).good(), "Result exists");
        if (doctest::detail::runAll() != 0)
            return 1;
        if (argc == 3) {
            const auto [vertices, faces] = check_real_surface(argv[1]);
            std::ofstream output(argv[2]);
            require(output.good(), "Cannot open result file");
            write_summary(output, vertices, faces);
        } else {
            write_summary(std::cout, 0, 0);
        }
    } catch (const std::exception& error) {
        std::cerr << "SLIM regression: " << error.what() << '\n';
        return 1;
    }
}
