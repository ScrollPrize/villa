#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include "vc_tomography/calibration/solve.hpp"
inline nanobind::dict fit_summary_dict(const vc_tomography::calibration::FitSummary& s) {
    nanobind::dict d;
    d["cost"] = s.cost;
    d["iterations"] = s.iterations;
    d["evaluations"] = s.evaluations;
    d["usable"] = s.usable;
    d["termination"] = s.termination;
    d["message"] = s.message;
    d["solver_version"] = s.solver_version;
    return d;
}
