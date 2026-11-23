#pragma once
#include "options.hpp"
#include <vc_reader/thinning.hpp>

namespace vcr::transforms {
enum class Morph { Dilate, Erode, Open, Close };
// ball: Euclidean distance <= radius. cross: L1 <= radius, SciPy's default
// structure iterated radius times. cube: Chebyshev <= radius, SciPy's full
// connectivity iterated. repeated_cross: the 3D six-connected cross with each
// cell expanded to a radius^3 block.
enum class MorphFootprint { Ball, Cross, Cube, RepeatedCross };
struct Morphology { Morph operation = Morph::Close; Scalar radius = 1.; std::vector<std::string> on; bool neutral_border = false; MorphFootprint footprint = MorphFootprint::Ball; double p = 1; };
struct RandomBinaryOperator { std::string on = "image"; std::vector<std::size_t> channels; Scalar radius = Uniform{1, 10}; std::vector<Morph> operations{Morph::Dilate, Morph::Erode, Morph::Open, Morph::Close}; double p_channel = 1, p = 1; };
struct RemoveComponents { std::string on = "image"; std::vector<std::size_t> channels; double max_fraction = .25, p_fill = .25, p_channel = 1, p = 1; bool full_connectivity = true; };
struct ReplaceLabel { std::string on = "mask"; double value = -1, replacement = 0, p = 1; std::vector<std::size_t> channels; };
struct Regions { std::string on = "mask", output; std::vector<std::vector<double>> labels; std::size_t channel = 0; double p = 1; };
struct OneHot { std::string on = "mask", output = "onehot", append_to; std::vector<double> labels; std::size_t channel = 0; bool remove_source = false; double p = 1; };
struct MaskImage { std::string on = "image", mask = "mask"; std::size_t mask_channel = 0; std::vector<std::size_t> channels; double fill = 0, p = 1; };
struct DeepSupervision { std::string on = "mask", output; std::vector<std::vector<double>> scales; double p = 1; };
struct ToPseudo2D { double p = 1; };
struct FromPseudo2D { double p = 1; };
// connectivity: 4/8 (2D) or 6/18/26 (3D); 0 for full (8 or 26).
struct ConnectedComponents { std::vector<std::string> on; std::string suffix="_components"; int connectivity=0; double p=1; };
struct RidgeTrace { std::vector<std::string> on; std::string suffix="_traces"; double p=1; };
struct MedialSurface { thinning::Algorithm method=thinning::Algorithm::ZhangSuen; bool preserve_border=false; std::vector<std::string> on; std::string suffix = "_skel"; bool tube = true, open = false, close = true; std::map<std::string, double> ignore; double p = 1; };
} // namespace vcr::transforms
