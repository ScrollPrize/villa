#include "FiberConstraintExtraction.hpp"
#include "vc/core/util/CudaSampling.hpp"
#include <exception>
#include <iostream>

int main(int argc, char** argv)
{
    vc::cuda::applyToolOptions(argc, argv);
    try {
        return runFiberConstraintExtraction(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "vc_fiber_constraints: " << error.what() << '\n';
        return 1;
    }
}
