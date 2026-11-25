#include "common.hpp"
#include "rules.hpp"
#include <array>
namespace vcr::thinning {
Algorithm algorithmFromName(std::string_view name) {
    if(name=="zhang_suen")return Algorithm::ZhangSuen;
    if(name=="guo_hall")return Algorithm::GuoHall;
    if(name=="zhang_suen_guo_hall")return Algorithm::ZhangSuenGuoHall;
    throw std::invalid_argument("method must be zhang_suen, guo_hall, or zhang_suen_guo_hall");
}
namespace {
void thin(std::uint8_t* image, std::size_t rows, std::size_t cols, Algorithm algorithm, bool preserveBorder) {
    std::array<std::array<std::uint8_t,256>,2> lut{};
    for(int phase=0;phase<2;++phase) for(unsigned c=0;c<256;++c) lut[phase][c]=detail::remove(c,phase,algorithm);
    std::vector<std::size_t> active;
    for(std::size_t y=0;y<rows;++y) for(std::size_t x=0;x<cols;++x)
        if(image[y*cols+x] && !(preserveBorder && (!y || !x || y+1==rows || x+1==cols))) active.push_back(y*cols+x);
    std::vector<std::uint8_t> marks(active.size());
    bool changed=true;
    while(changed) {
        changed=false;
        for(int phase=0;phase<2;++phase) {
            for(std::size_t j=0;j<active.size();++j) marks[j]=lut[phase][detail::neighbors(image,active[j],rows,cols)];
            std::size_t kept=0;
            for(std::size_t j=0;j<active.size();++j) {
                if(marks[j]) {image[active[j]]=0;changed=true;}
                else active[kept++]=active[j];
            }
            active.resize(kept);
        }
    }
}
}
void skeleton(const std::uint8_t* input,std::uint8_t* output,std::size_t rows,std::size_t cols,
              std::size_t slices,Algorithm algorithm,bool preserveBorder) {
    const auto n=detail::validate(input,rows,cols,slices);
    detail::validateAlgorithm(algorithm);
    if(n && !output) throw std::invalid_argument("null skeleton output");
    for(std::size_t i=0;i<n;++i) output[i]=input[i]?255:0;
    if(!n) return;
    for(std::size_t z=0;z<slices;++z) {
        auto* image=output+z*rows*cols;
        thin(image,rows,cols,algorithm==Algorithm::ZhangSuenGuoHall?Algorithm::ZhangSuen:algorithm,preserveBorder);
        if(algorithm==Algorithm::ZhangSuenGuoHall) thin(image,rows,cols,Algorithm::GuoHall,preserveBorder);
    }
}
}
