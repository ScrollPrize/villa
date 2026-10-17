#pragma once
#include "resident.cuh"
#include "../fourier.hpp"
namespace vcr::augment::cuda {
template<class T> void fft_volume(Buffer<volume_detail::ComplexT<T>>& data,const volume_detail::Grid3& grid,bool inverse);
template<class T> void to_complex(View<T>,Buffer<volume_detail::ComplexT<T>>&);
template<class T> void from_complex(const Buffer<volume_detail::ComplexT<T>>&,View<T>);
}
