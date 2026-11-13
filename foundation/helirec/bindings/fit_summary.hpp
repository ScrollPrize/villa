#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include "helirec/calibration/solve.hpp"
inline nanobind::dict fit_summary_dict(const helirec::calibration::FitSummary& s) {
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
