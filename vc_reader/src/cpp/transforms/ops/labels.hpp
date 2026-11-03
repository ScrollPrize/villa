#pragma once
#include "options.hpp"

namespace vcr::transforms {
enum class Morph { Dilate, Erode, Open, Close };
struct Morphology { Morph operation = Morph::Close; Scalar radius = 1.; std::vector<std::string> on; bool neutral_border = false, repeated_cross = false; double p = 1; };
struct RandomBinaryOperator { std::string on = "image"; std::vector<std::size_t> channels; Scalar radius = Uniform{1, 10}; std::vector<Morph> operations{Morph::Dilate, Morph::Erode, Morph::Open, Morph::Close}; double p_channel = 1, p = 1; };
struct RemoveComponents { std::string on = "image"; std::vector<std::size_t> channels; double max_fraction = .25, p_fill = .25, p_channel = 1, p = 1; bool full_connectivity = true; };
struct ReplaceLabel { std::string on = "mask"; double value = -1, replacement = 0, p = 1; std::vector<std::size_t> channels; };
struct Regions { std::string on = "mask", output; std::vector<std::vector<double>> labels; std::size_t channel = 0; double p = 1; };
struct OneHot { std::string on = "mask", output = "onehot", append_to; std::vector<double> labels; std::size_t channel = 0; bool remove_source = false; double p = 1; };
struct MaskImage { std::string on = "image", mask = "mask"; std::size_t mask_channel = 0; std::vector<std::size_t> channels; double fill = 0, p = 1; };
struct DeepSupervision { std::string on = "mask", output; std::vector<std::vector<double>> scales; double p = 1; };
struct ToPseudo2D { double p = 1; };
struct FromPseudo2D { double p = 1; };
struct MedialSurface { std::vector<std::string> on; std::string suffix = "_skel"; bool tube = true, open = false, close = true; std::map<std::string, double> ignore; double p = 1; };
} // namespace vcr::transforms
