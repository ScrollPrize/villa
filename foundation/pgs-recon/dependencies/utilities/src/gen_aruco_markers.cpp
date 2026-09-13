#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <optional>
#include <set>
#include <string_view>

#include <boost/program_options.hpp>
#include <educelab/core/utils/Iteration.hpp>
#include <educelab/core/utils/String.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

namespace ar = cv::aruco;
namespace el = educelab;
namespace po = boost::program_options;
namespace fs = std::filesystem;

enum TIFFUnit {
  NONE = 1,      /* no meaningful units */
  INCH = 2,      /* english */
  CENTIMETER = 3 /* metric */
};

enum OutType {
  IMAGE = 0x01,
  SHEET = 0x02,
  BOTH = IMAGE | SHEET,
};

enum PaperSize { LETTER, A4 };

using SizeMap = std::map<PaperSize, cv::Size2d>;
SizeMap PAPER_SIZES{
  {LETTER, {8.5, 11}},
  {A4, {21, 29.7}}
};

// clang-format off
struct SheetParams {
  cv::Size2d size{21, 29.7};
  double margin{1.0};
  double padding{0.25};
  int res{120};
  bool cropMarks{true};
};
// clang-format on

// Colors in BGR order
using Color = cv::Vec3b;
using ColorMap = std::map<std::string, Color>;
static const Color WHITE{255, 255, 255};
static const Color BLACK{0, 0, 0};
static const Color RED{0, 0, 255};
static const Color GREEN{0, 255, 0};
static const Color BLUE{255, 0, 0};
static const Color CYAN{255, 255, 0};
static const Color YELLOW{0, 255, 255};
static const Color MAGENTA{255, 0, 255};
static const Color ORANGE{255, 128, 0};
static const Color PURPLE{128, 0, 255};
static const Color JASPER{59, 87, 213};
static const Color LIGHT_GRAY{200, 200, 200};
static const Color GRAY{127, 127, 127};
static const Color DARK_GRAY{68, 68, 68};
// clang-format off
ColorMap COLOR_MAP {
    {"white", WHITE},
    {"black", BLACK},
    {"red", RED},
    {"green", GREEN},
    {"blue", BLUE},
    {"cyan", CYAN},
    {"magenta", MAGENTA},
    {"yellow", YELLOW},
    {"orange", ORANGE},
    {"purple", PURPLE},
    {"jasper", JASPER},
    {"light-gray", LIGHT_GRAY},
    {"gray", GRAY},
    {"dark-gray", DARK_GRAY},
  };
// clang-format on

// Parser
auto RGBStringToBGR(std::string_view rgbStr) -> std::optional<cv::Vec3b> {
  // split the string
  const auto parts = el::split(rgbStr, ",");

  // validate size
  if (parts.size() != 3) {
    return std::nullopt;
  }

  // retrieve the values
  cv::Vec3b result;
  try {
    result[2] = el::to_numeric<std::uint8_t>(parts[0]);
    result[1] = el::to_numeric<std::uint8_t>(parts[1]);
    result[0] = el::to_numeric<std::uint8_t>(parts[2]);
  } catch (const std::exception &e) {
    return std::nullopt;
  }
  return result;
}

using MarkerList = std::vector<std::pair<std::string, cv::Mat>>;

auto PadAndLabel(const cv::Mat &marker, const std::string &label, int padding,
                 bool cropMarks = true) -> cv::Mat {
  // Width is easy
  auto w = marker.cols + 2 * padding;

  // Build the height
  // pad top, image, pad below image
  auto h = padding + marker.rows + padding;

  // Add the text size
  constexpr auto font = cv::FONT_HERSHEY_SIMPLEX;
  constexpr double scale{1.};
  constexpr int thick{1};
  int base{0};
  cv::Size textSize{0, 0};
  int textBot{0};
  if (not label.empty()) {
    textSize = cv::getTextSize(label, font, scale, thick, &base);
    base += thick;
    h += textSize.height;
    if (textSize.width > w) {
      w = textSize.width + 2 * padding;
    }
    textBot = h;
    // pad below text
    h += padding;
  }

  // Build the padded image
  cv::Mat out(h, w, marker.type(), cv::Scalar::all(255));

  // Copy the marker
  auto iy = padding;
  auto ix = (w - marker.cols) / 2;
  marker.copyTo(out({ix, iy, marker.cols, marker.rows}));

  // Draw the crop marks
  if (cropMarks) {
    const int mPad = padding / 2;
    const cv::Scalar mColor = cv::Scalar::all(0);
    // Horizontal left
    cv::line(out, {0, iy - 1}, {ix - mPad, iy - 1}, mColor, 1);
    cv::line(out, {0, iy + marker.rows + 1}, {ix - mPad, iy + marker.rows + 1},
             mColor, 1);
    // Horizontal right
    cv::line(out, {ix + marker.cols + mPad, iy - 1}, {out.cols, iy - 1}, mColor,
             1);
    cv::line(out, {ix + marker.cols + mPad, iy + marker.rows + 1},
             {out.cols, iy + marker.rows + 1}, mColor, 1);
    // Vertical top
    cv::line(out, {ix - 1, 0}, {ix - 1, iy - mPad}, mColor, 1);
    cv::line(out, {ix + marker.cols + 1, 0}, {ix + marker.cols + 1, iy - mPad},
             mColor, 1);
    // Vertical bottom
    cv::line(out, {ix - 1, iy + marker.rows + mPad},
             {ix - 1, 2 * iy + marker.rows}, mColor, 1);
    cv::line(out, {ix + marker.cols + 1, iy + marker.rows + mPad},
             {ix + marker.cols + 1, 2 * iy + marker.rows}, mColor, 1);
  }

  // Write the text
  if (not label.empty()) {
    iy = textBot;
    ix = (w - textSize.width) / 2;
    cv::putText(out, label, {ix, iy}, font, scale, thick);
  }

  return out;
}

auto GenerateContactSheets(const MarkerList &markers, const SheetParams &params)
    -> std::vector<cv::Mat> {
  const int h = static_cast<int>(std::round(params.size.height * params.res));
  const int w = static_cast<int>(std::round(params.size.width * params.res));
  const int m = static_cast<int>(std::round(params.margin * params.res));
  const int p = static_cast<int>(std::round(params.padding * params.res));
  const bool c = params.cropMarks;

  // Pad and label all the markers
  std::vector<cv::Mat> padded;
  std::transform(markers.begin(), markers.end(), std::back_inserter(padded),
                 [&p, &c](const auto &pair) {
                   return PadAndLabel(pair.second, pair.first, p, c);
                 });

  // TODO: Sanity check that the sheet can hold at least one marker

  // Set up the first sheet
  const auto imgType = padded.front().type();
  std::vector<cv::Mat> sheets;
  cv::Mat sheet(h, w, imgType, cv::Scalar::all(255));
  int oy{m};
  int ox{m};
  for (const auto &marker : padded) {
    // If we're at the end of the row, increment
    if (ox + marker.cols > w - m) {
      oy += marker.rows;
      ox = m;
    }
    // If we're at the end of the sheet, make a new one
    if (oy + marker.rows > h - m) {
      sheets.push_back(sheet);
      sheet = cv::Mat(h, w, imgType, cv::Scalar::all(255));
      oy = ox = m;
    }
    marker.copyTo(sheet({ox, oy, marker.cols, marker.rows}));
    ox += marker.cols;
  }
  sheets.push_back(sheet);

  return sheets;
}

auto main(int argc, char *argv[]) -> int {
  // clang-format off
    po::options_description parser("options");
    parser.add_options()
      ("help,h", "print help message")
      ("output-dir,o", po::value<std::string>(), "Output directory")
      ("output-type,t", po::value<std::string>()->default_value("image"), "Output type: image, sheet, both")
      ("id", po::value<std::vector<int>>()->multitoken(), "Marker ID number(s)")
      ("id-range", po::value<std::vector<int>>()->multitoken(), "Marker ID range: start end {step}")
      ("image-prefix,p", po::value<std::string>()->default_value("marker_"), "Marker output prefix")
      ("sheet-prefix,s", po::value<std::string>()->default_value("markers_"), "Contact sheet output prefix")
      ("marker-size,m", po::value<double>()->default_value(3), "Marker size in world units")
      ("marker-unit,u", po::value<std::string>()->default_value("cm"), "World unit: cm, in[ch]")
      ("resolution,r", po::value<int>()->default_value(120), "Image resolution in pixels/unit")
      ("fg-color", po::value<std::string>(), "The color for the marker foreground (Default: white). A predefined color (red, yellow, blue, etc.) or an 8-bit RGB value in the format: \'R,G,B\'")
      ("bg-color", po::value<std::string>(), "The color for the marker background (Default: black). A predefined color (red, yellow, blue, etc.) or an 8-bit RGB value in the format: \'R,G,B\'")
      ("quiet-color", po::value<std::string>(), "The color for the quiet zone (Default: same as foreground). A predefined color (red, yellow, blue, etc.) or an 8-bit RGB value in the format: \'R,G,B\'")
      ("quiet-size", po::value<float>()->default_value(0.1), "Size of the quiet zone as a percentage of the marker size")
      ("crop-marks", po::value<bool>()->default_value(true), "Add crop marks to contact sheets")
    ;
    // clang-format on

    po::variables_map args;
    po::store(po::parse_command_line(argc, argv, parser), args);
    if (argc == 1 or args.count("help") > 0) {
      std::cout << parser << "\n";
      return EXIT_FAILURE;
    }
  po::notify(args);

  // Get dictionary
  const auto dict = ar::getPredefinedDictionary(ar::DICT_ARUCO_ORIGINAL);

  // Set up image params
  const auto sizeWorld = args["marker-size"].as<double>();
  const auto res = args["resolution"].as<int>();
  const auto sizePixel = static_cast<int>(std::round(sizeWorld * res));
  // Set up world units
  int unit{CENTIMETER};
  auto unitStr = el::to_lower_copy(args["marker-unit"].as<std::string>());
  if (unitStr == "cm") {
    unit = CENTIMETER;
  } else if (unitStr == "in" or unitStr == "inch") {
    unit = INCH;
  } else {
    std::cout << "WARNING: Unknown marker unit: " << unitStr
              << ". Falling back to centimeter.\n";
  }
  unitStr = ", " + std::to_string(res) + "px/" + unitStr;
  std::vector<int> params{cv::IMWRITE_TIFF_RESUNIT, unit,
                          cv::IMWRITE_TIFF_XDPI,    res,
                          cv::IMWRITE_TIFF_YDPI,    res};

  //  Collect the IDs list
  std::set<int> ids;
  if (args.count("id") > 0) {
    auto individual = args["id"].as<std::vector<int>>();
    ids.insert(individual.begin(), individual.end());
  }
  if (args.count("id-range") > 0) {
    const auto rParams = args["id-range"].as<std::vector<int>>();
    if (rParams.size() < 2) {
      std::cout << "WARNING: --id-range requires two arguments. Ignoring.\n";
    } else if (rParams.size() > 3) {
      std::cout
          << "WARNING: --id-range only accepts three arguments. Ignoring.\n";
    } else {
      auto start = rParams[0];
      auto end = rParams[1];
      auto step = rParams.size() > 2 ? rParams[2] : 1;
      auto r = el::range(start, end, step);
      ids.insert(r.begin(), r.end());
    }
  }
  if (ids.empty()) {
    std::cout << "ERROR: Must provide --id and/or --id-range\n";
    return EXIT_FAILURE;
  }

  // Set up output directory
  fs::path outDir = fs::current_path();
  if (args.count("output-dir") > 0) {
    outDir = args["output-dir"].as<std::string>();
  }
  fs::create_directories(outDir);

  // Set up output types
  std::vector<std::string_view> typeOpts = {"image", "sheet", "both"};
  const auto typeStr = el::to_lower_copy(args["output-type"].as<std::string>());
  if (std::find(typeOpts.begin(), typeOpts.end(), typeStr) == typeOpts.end()) {
    std::cout << "ERROR: Unrecognized output type: " + typeStr + "\n";
    return EXIT_FAILURE;
  }
  auto saveImage = typeStr == "image" or typeStr == "both";
  auto saveSheet = typeStr == "sheet" or typeStr == "both";

  // Set up color replacement
  cv::Vec3b fgColor{WHITE};
  cv::Vec3b bgColor{BLACK};
  if (args.count("fg-color") > 0) {
    auto colorStr = el::to_lower_copy(args["fg-color"].as<std::string>());
    if (COLOR_MAP.count(colorStr) > 0) {
      fgColor = COLOR_MAP[colorStr];
    } else {
      if (const auto c = RGBStringToBGR(colorStr)) {
        fgColor = c.value();
      } else {
        std::cout << "WARNING: Unrecognized color: " + colorStr + "\n";
      }
    }
  }
  if (args.count("bg-color") > 0) {
    auto colorStr = el::to_lower_copy(args["bg-color"].as<std::string>());
    if (COLOR_MAP.count(colorStr) > 0) {
      bgColor = COLOR_MAP[colorStr];
    } else {
      if (const auto c = RGBStringToBGR(colorStr)) {
        bgColor = c.value();
      } else {
        std::cout << "WARNING: Unrecognized color: " + colorStr + "\n";
      }
    }
  }
  // Get border color (default: same as foreground)
  cv::Vec3b quietColor = fgColor;
  if (args.count("quiet-color") > 0) {
    auto colorStr = el::to_lower_copy(args["quiet-color"].as<std::string>());
    if (COLOR_MAP.count(colorStr) > 0) {
      quietColor = COLOR_MAP[colorStr];
    } else {
      if (const auto c = RGBStringToBGR(colorStr)) {
        quietColor = c.value();
      } else {
        std::cout << "WARNING: Unrecognized color: " + colorStr + "\n";
      }
    }
  }
  auto replaceColor = fgColor != WHITE or bgColor != BLACK;
  auto needColorOut = replaceColor or quietColor != WHITE;

  // File name setup
  auto imgPrefix = args["image-prefix"].as<std::string>();
  int pad{4};
  const std::string ext{".tif"};

  // Generate (and optionally save) markers
  MarkerList imgs;
  std::cout << "Generating " << ids.size() << " markers (" << sizePixel << "x"
            << sizePixel << unitStr << ")\n";

  for (const auto &id : ids) {
    // Generate the marker
    cv::Mat marker;
    ar::generateImageMarker(dict, id, sizePixel, marker);

    // Replace colors
    if (needColorOut) {
      cv::cvtColor(marker, marker, cv::COLOR_GRAY2BGR);
    }
    if (replaceColor) {
      std::for_each(marker.begin<cv::Vec3b>(), marker.end<cv::Vec3b>(),
                    [&fgColor, &bgColor](cv::Vec3b &c) {
                      if (c == WHITE) {
                        c = fgColor;
                      } else if (c == BLACK) {
                        c = bgColor;
                      }
                    });
    }

    // Add quiet zone to image
    if (const auto &qRate = args["quiet-size"].as<float>(); qRate > 0) {
      int quietPix = std::max(marker.rows, marker.cols) * qRate;
      auto imgSize = marker.size() + cv::Size{2 * quietPix, 2 * quietPix};
      cv::Mat quietImg(imgSize, marker.type(),
                       cv::Scalar(quietColor[0], quietColor[1], quietColor[2]));
      marker.copyTo(quietImg({quietPix, quietPix, marker.cols, marker.rows}));
      marker = quietImg;
    }

    // Pad the ID
    const auto idStr = el::to_padded_string(id, pad);

    // Save the image if requested
    if (saveImage) {
      const auto f = imgPrefix + idStr + ext;
      cv::imwrite(outDir / f, marker, params);
    }

    // Store the image if we're also going to build a contact sheet
    if (saveSheet) {
      imgs.emplace_back(idStr, marker);
    }
  }

  // Generate the contact sheet
  auto sheetPrefix = args["sheet-prefix"].as<std::string>();
  if (saveSheet) {
    std::cout << "Generating contact sheet\n";

    SheetParams p;
    p.res = res;
    p.cropMarks = args["crop-marks"].as<bool>();
    if (unit == CENTIMETER) {
      p.size = PAPER_SIZES[A4];
      p.margin = 1.0;
      p.padding = 0.25;
    } else {
      p.size = PAPER_SIZES[LETTER];
      p.margin = 0.4;
      p.padding = 0.1;
    }

    auto sheets = GenerateContactSheets(imgs, p);
    pad = static_cast<int>(std::to_string(sheets.size()).size());
    for (const auto &[id, sheet] : el::enumerate(sheets)) {
      const auto idStr = el::to_padded_string(id, pad);
      const auto f = sheetPrefix + idStr + ext;
      cv::imwrite(outDir / f, sheet, params);
    }
  }
  std::cout << "Done.\n";
  return EXIT_SUCCESS;
}