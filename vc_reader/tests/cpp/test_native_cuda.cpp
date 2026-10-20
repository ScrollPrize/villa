#include <vc_reader/transforms.hpp>
#include <vc_reader/cuda.hpp>
#ifdef VCR_TEST_READER
#include <vc_reader/cuda_reader.hpp>
#endif
#include <iostream>

int main()
{
    using namespace vcr::transforms;
    if (!cuda::available()) return 77;
    try {
        cudaStream_t stream;
        vcr::device::check(cudaStreamCreate(&stream));
        struct Cleanup { cudaStream_t stream; ~Cleanup() { cudaStreamDestroy(stream); } } cleanup{stream};
        vcr::device::StreamScope scope(stream);
        const Target image{};
        Sample input;
        input["image"] = {Array({1,3,4}, 'f', 4), image};
        auto* values = static_cast<float*>(input.at("image").array.data());
        std::iota(values, values + 12, 1.f);
        BrightnessMultiply multiply;
        multiply.factor = 2.;
        const std::vector<Op> operations{multiply, Flip{"x", 1., 1.}};
        Pipeline cpu(operations, {{"image",image}}, true, "cpu");
        Pipeline gpu(operations, {{"image",image}}, true, "cuda");
        auto expected = cpu.run(input, 42, true, false);
        auto result = gpu.run(input, 42, true, false);
        require(std::memcmp(expected.sample.at("image").array.data(), result.sample.at("image").array.data(),
                            12*sizeof(float)) == 0, "CUDA transform values differ from CPU");
        auto replay = gpu.replay(input, load_plan(gpu.program(), save_plan(result.plan)));
        require(std::memcmp(expected.sample.at("image").array.data(), replay.sample.at("image").array.data(),
                            12*sizeof(float)) == 0, "CUDA replay values differ from CPU");
#ifdef VCR_TEST_READER
        std::vector<uint8_t> patch(109*109*109);
        for (size_t i=0; i<patch.size(); ++i) patch[i] = uint8_t(120+70*std::sin((int(i%109)-54)*.3));
        const float center[]{48,48,48,1};
        float normal[4]{}, actual[4]{};
        vcr::sheet_normal::cpu(patch.data(), center, normal, 1);
        auto device = vcr::device::allocate(patch.size(), 0);
        vcr::device::check(cudaMemcpyAsync(device->ptr, patch.data(), patch.size(), cudaMemcpyHostToDevice, stream));
        auto output = vcr::sheet_normal::cuda(device->ptr, false, center, 1, 0);
        vcr::device::check(cudaMemcpyAsync(actual, output->ptr, sizeof(actual), cudaMemcpyDeviceToHost, stream));
        vcr::device::complete();
        require(std::abs(actual[0]*normal[0]+actual[1]*normal[1]+actual[2]*normal[2]) > .99999f,
                "native CPU/CUDA normal alignment");
        require(std::abs(actual[3]-normal[3]) < 2e-5f, "native CPU/CUDA eigengap");
#endif
        std::cout << "Native CUDA transforms, replay, streams and normals passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
