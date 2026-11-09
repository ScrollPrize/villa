#include <string>

#include "helirec/fourier.hpp"

namespace helirec {

PadMode pad_mode_from_string(const std::string& s) {
    if (s == "zero" || s == "constant" || s == "zeros") return PadMode::Zero;
    if (s == "edge") return PadMode::Edge;
    if (s == "reflect") return PadMode::Reflect;
    if (s == "symmetric" || s == "mirror") return PadMode::Symmetric;
    if (s == "wrap" || s == "periodic") return PadMode::Wrap;
    throw Error("unknown pad mode '" + s + "'");
}

void validate(const FourierFilterSpec& s) {
    if (s.ndim != 1 && s.ndim != 2) throw Error("FourierFilterSpec.ndim must be 1 or 2");
    if (s.h <= 0 || s.w <= 0 || s.pw < s.w) throw Error("FourierFilterSpec: bad frame/padded width");
    if (s.pad_left < 0 || s.pad_left + s.w > s.pw) throw Error("FourierFilterSpec: pad_left out of range");
    if (s.ndim == 2) {
        if (s.ph < s.h) throw Error("FourierFilterSpec: padded height < height");
        if (s.pad_top < 0 || s.pad_top + s.h > s.ph) throw Error("FourierFilterSpec: pad_top out of range");
    }
    const std::size_t nf = static_cast<std::size_t>(s.pw / 2 + 1) * (s.ndim == 2 ? s.ph : 1);
    if (s.filter.size() != nf)
        throw Error("FourierFilterSpec: filter has " + std::to_string(s.filter.size()) +
                    " values, expected " + std::to_string(nf));
    if (!s.filter_imag.empty() && s.filter_imag.size() != nf)
        throw Error("FourierFilterSpec: filter_imag must be empty or match filter");
}

}  // namespace helirec
