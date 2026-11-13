#pragma once
#include <cmath>
namespace helirec::calibration::fitting {
template <class S> S sinc_pi(const S& a) {
    using std::sin;
    const S z = S(M_PI)*a;
    // Stable value and derivative at zero, including automatic differentiation.
    return z*z < S(1e-12) ? S(1)-z*z/S(6)+z*z*z*z/S(120) : sin(z)/z;
}

}
