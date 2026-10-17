#pragma once
#include "../core.hpp"
namespace vcr::augment::cuda {
struct DeviceInput { void* data; Shape shape; char kind; std::size_t bytes; int device; };
std::shared_ptr<void> begin_device(const std::map<std::string,DeviceInput>&, Sample&, bool inplace);
std::map<std::string,DeviceInput> device_outputs(void*);
void finish_device(void*);
}
