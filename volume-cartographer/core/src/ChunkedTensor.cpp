#include "vc/core/types/ChunkedTensor.hpp"

#include <iostream>

void print_accessor_stats()
{
    std::cout << "acc miss/total " << miss << " " << total << " " << double(miss)/total << std::endl;
    std::cout << "chunk compute overhead/total " << chunk_compute_collisions << " " << chunk_compute_total << " " << double(chunk_compute_collisions)/chunk_compute_total << std::endl;
}
