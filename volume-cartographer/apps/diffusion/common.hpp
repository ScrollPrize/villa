#pragma once

#include <opencv2/core/mat.hpp>
#include <vc/core/types/VcDataset.hpp>
#include <vc/core/util/Slicing.hpp>

#include <filesystem>
#include <optional>

namespace boost::program_options { class variables_map; }
namespace po = boost::program_options;
namespace fs = std::filesystem;

struct discrete_options;
struct continous_options;

void setup_cli_and_run(int argc, char** argv);