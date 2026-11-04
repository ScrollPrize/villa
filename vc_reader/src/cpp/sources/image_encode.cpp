#include "vc_reader/image_write.hpp"
#include "vc_reader/zarr.hpp"
#include <png.h>
#include <cstdio>
#include <jpeglib.h>
#include <csetjmp>
#include <cstdlib>
#include <algorithm>
#include <cctype>
#include <limits>
#include <memory>
#include <stdexcept>
namespace vcr::sources {
namespace {
void check(ImageView image, bool jpeg) {
    if (!image.data || image.dtype != "|u1" || image.rows <= 0 || image.cols <= 0 ||
        image.rows > 0x7fffffff || image.cols > 0x7fffffff ||
        image.channels < 1 || image.channels > 4 || (jpeg && image.channels != 1 && image.channels != 3))
        throw std::invalid_argument("PNG/JPEG requires a positive uint8 gray or RGB image (PNG also allows alpha)");
    if (std::uint64_t(image.rows)*image.cols > std::numeric_limits<std::size_t>::max()/image.channels)
        throw std::invalid_argument("image is too large");
}
struct JpegState {
    jpeg_compress_struct encoder{};
    struct Error : jpeg_error_mgr { std::jmp_buf jump; } error;
    unsigned char* bytes = nullptr;
    unsigned long size = 0;
    ~JpegState() { jpeg_destroy_compress(&encoder); std::free(bytes); }
};
}
std::vector<std::uint8_t> encodePng(ImageView image) {
    check(image,false);
    png_image png{};
    png.version=PNG_IMAGE_VERSION;png.width=png_uint_32(image.cols);png.height=png_uint_32(image.rows);
    const png_uint_32 formats[]{PNG_FORMAT_GRAY,PNG_FORMAT_GA,PNG_FORMAT_RGB,PNG_FORMAT_RGBA};
    png.format=formats[image.channels-1];
    png_alloc_size_t size=0;
    if (!png_image_write_to_memory(&png,nullptr,&size,0,image.data,0,nullptr)) {
        std::string error=png.message;png_image_free(&png);throw std::runtime_error(error);
    }
    std::vector<std::uint8_t> bytes(size);
    if (!png_image_write_to_memory(&png,bytes.data(),&size,0,image.data,0,nullptr)) {
        std::string error=png.message;png_image_free(&png);throw std::runtime_error(error);
    }
    png_image_free(&png);bytes.resize(size);return bytes;
}
std::vector<std::uint8_t> encodeJpeg(ImageView image,int quality) {
    check(image,true);
    if(quality<0 || quality>100)throw std::invalid_argument("JPEG quality must be in [0,100]");
    const auto owner = std::make_unique<JpegState>();
    auto& state = *owner;
    auto& encoder=state.encoder;
    encoder.err=jpeg_std_error(&state.error);
    state.error.error_exit=[](j_common_ptr p) { std::longjmp(static_cast<JpegState::Error*>(p->err)->jump,1); };
    if(setjmp(state.error.jump))throw std::runtime_error("cannot encode JPEG");
    jpeg_create_compress(&encoder);
    jpeg_mem_dest(&encoder,&state.bytes,&state.size);
    encoder.image_width=JDIMENSION(image.cols);encoder.image_height=JDIMENSION(image.rows);
    encoder.input_components=int(image.channels);encoder.in_color_space=image.channels==1?JCS_GRAYSCALE:JCS_RGB;
    jpeg_set_defaults(&encoder);jpeg_set_quality(&encoder,quality,TRUE);jpeg_start_compress(&encoder,TRUE);
    while(encoder.next_scanline<encoder.image_height) {
        auto* row=const_cast<JSAMPLE*>(static_cast<const JSAMPLE*>(image.data)+std::size_t(encoder.next_scanline)*image.cols*image.channels);
        jpeg_write_scanlines(&encoder,&row,1);
    }
    jpeg_finish_compress(&encoder);
    return {state.bytes,state.bytes+state.size};
}
void writeImage(const std::filesystem::path& path,ImageView image,int quality) {
    auto suffix=path.extension().string();
    std::transform(suffix.begin(),suffix.end(),suffix.begin(),[](unsigned char c){return char(std::tolower(c));});
    if(suffix==".tif" || suffix==".tiff") {writeTiff(path,image);return;}
    const auto bytes=suffix==".png"?encodePng(image):suffix==".jpg" || suffix==".jpeg"?encodeJpeg(image,quality):
        throw std::invalid_argument("image extension must be png, jpg, jpeg, tif or tiff");
    zarr::atomicWrite(path,bytes.data(),bytes.size());
}
}
