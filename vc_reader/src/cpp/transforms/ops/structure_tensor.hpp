#pragma once
#include "options.hpp"

namespace vcr::transforms {
enum class Derivative { Pavel, Gaussian };
enum class StructureField { Components, Normal, Tangent };
// pavel: vesuvius StructureTensorComputer (Gaussian presmoothing radius
// int(3 sigma), Holoborodko derivative, component smoothing int(3 rho)).
// gaussian: SciPy Gaussian derivative (radius int(truncate*sigma + .5)) and
// Gaussian integration, as tracers' ct_structure_tensor. at: evaluate one
// tensor with Gaussian weights normalized over the whole grid.
struct StructureTensor {
    std::string on = "image"; std::size_t channel = 0; std::string output;
    double sigma = 1, rho = 1.5, truncate = 3, scale = 1;
    Derivative derivative = Derivative::Pavel; Boundary boundary = Boundary::Zero;
    StructureField field = StructureField::Components; std::string components = "zyx", confidence;
    std::vector<double> at;
    double p = 1;
};
} // namespace vcr::transforms
