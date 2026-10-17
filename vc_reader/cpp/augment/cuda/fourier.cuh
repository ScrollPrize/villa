#pragma once
#include "resident.cuh"
#include "../fourier.hpp"
namespace vcr::augment::cuda {
void fft_volume(Buffer<volume_detail::Complex>& data,const volume_detail::Grid3& grid,bool inverse);
void to_complex(View<double>,Buffer<volume_detail::Complex>&);
void from_complex(const Buffer<volume_detail::Complex>&,View<double>);
}
