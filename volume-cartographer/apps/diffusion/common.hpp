#pragma once

#include <opencv2/opencv.hpp>
#include <z5/factory.hxx>
#include <boost/program_options.hpp>
#include <vc/core/util/Slicing.hpp>

#include <filesystem>

namespace po = boost::program_options;
namespace fs = std::filesystem;

struct discrete_options;
struct continous_options;

void setup_cli_and_run(int argc, char** argv);