#include <cuda_runtime.h>
#include <type_traits>

static_assert(std::is_same_v<cudaError_t, enum cudaError>);

bool hip_available();

int main() {
    int cuda_version = 0;
    (void)cudaRuntimeGetVersion(&cuda_version);
    return hip_available() ? 0 : 77;
}
