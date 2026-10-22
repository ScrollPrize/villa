#include "FiberAfvConstraintBenchmark.hpp"
#include <exception>
#include <iostream>

int main(int argc, char** argv)
{
    try {
        return runFiberAfvConstraintBenchmark(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "vc_fiber_constraints: " << error.what() << '\n';
        return 1;
    }
}
