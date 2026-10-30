#include <vc_reader/transforms.hpp>
#include <vc_reader/cuda.hpp>
#ifdef VCR_TEST_READER
#include <vc_reader/cuda_reader.hpp>
#endif
#include <cstring>
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
        auto expected = cpu.run(input, 42, true);
        auto result = gpu.run(input, 42, true);
        require(std::memcmp(expected.sample.at("image").array.data(), result.sample.at("image").array.data(),
                            12*sizeof(float)) == 0, "CUDA transform values differ from CPU");
        auto replay = gpu.replay(input, load_plan(gpu.program(), save_plan(result.plan)));
        require(std::memcmp(expected.sample.at("image").array.data(), replay.sample.at("image").array.data(),
                            12*sizeof(float)) == 0, "CUDA replay values differ from CPU");
        // Device samples and batches, against the CPU run of each sample.
        {
            auto upload = [&](const void* data, std::size_t bytes) {
                auto storage = vcr::device::allocate(bytes, 0);
                vcr::device::check(cudaMemcpyAsync(storage->ptr, data, bytes, cudaMemcpyHostToDevice, stream));
                return storage;
            };
            auto download = [&](const void* data, std::size_t bytes) {
                std::vector<char> out(bytes);
                vcr::device::check(cudaMemcpyAsync(out.data(), data, bytes, cudaMemcpyDeviceToHost, stream));
                vcr::device::complete();
                return out;
            };
            const std::size_t count = 2, sample = 12*sizeof(float);
            std::vector<float> batch(24);
            std::iota(batch.begin(), batch.end(), 1.f);
            auto single = upload(batch.data(), sample);
            const auto run = cuda::run_device(gpu, {{"image", cuda::DeviceInput{single->ptr, {1,3,4}, 'f', 4, 0}}}, 42, false, false);
            const auto outputs = cuda::device_outputs(run.session.get());
            require(std::memcmp(download(outputs.at("image").data, sample).data(), expected.sample.at("image").array.data(), sample) == 0,
                    "device sample differs from CPU");

            auto stacked = upload(batch.data(), batch.size()*sizeof(float));
            std::vector<vcr::device::Storage> owned;
            const cuda::BatchAllocator allocate = [&](const Shape& shape, char kind, std::size_t bytes, int device) {
                owned.push_back(vcr::device::allocate(product(shape)*bytes, device));
                return cuda::DeviceInput{owned.back()->ptr, shape, kind, bytes, device};
            };
            const std::vector<std::uint64_t> seeds{42, 43};
            auto ran = cuda::run_batch(gpu, {{"image", cuda::DeviceInput{stacked->ptr, {count,1,3,4}, 'f', 4, 0}}}, seeds, {}, allocate);
            ran.outputs->check();
            const auto& array = ran.arrays.at("image");
            require(array.stacked && array.arrays.size() == 1 && array.arrays[0].shape == Shape({count,1,3,4}), "batch output is not stacked");
            const auto values = download(array.arrays[0].data, 2*sample);
            for (std::size_t b = 0; b < count; ++b) {
                Sample one;
                one["image"] = {Array({1,3,4}, 'f', 4), image};
                std::memcpy(one.at("image").array.data(), batch.data()+b*12, sample);
                const auto reference = cpu.run(one, seeds[b], false);
                require(std::memcmp(values.data()+b*sample, reference.sample.at("image").array.data(), sample) == 0, "batch sample differs from CPU");
            }
            bool refused = false;
            try { (void)cuda::run_batch(gpu, {{"image", cuda::DeviceInput{stacked->ptr, {count,1,3,4}, 'f', 4, 0}}}, {1}, {}, allocate); }
            catch (const std::invalid_argument&) { refused = true; }
            require(refused, "a batch with the wrong seed count must be refused");
        }
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
