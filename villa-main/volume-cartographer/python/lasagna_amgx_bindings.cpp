#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "vc/lasagna/LaplaceRank.hpp"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

namespace nb = nanobind;
using namespace nb::literals;

namespace {

nb::dict rankSnapPairs(nb::dict request, nb::object progressCallback = nb::none())
{
    nb::object json = nb::module_::import_("json");
    const std::string requestText = nb::cast<std::string>(json.attr("dumps")(request));
    std::string responseText;
    const bool hasCallback = !progressCallback.is_none();
    vc::lasagna::LaplaceRankProgressCallback callback;
    if (hasCallback) {
        callback = [progressCallback, json](const nlohmann::json& event) {
            nb::gil_scoped_acquire acquire;
            nb::object result = progressCallback(json.attr("loads")(event.dump()));
            if (!result.is_none() && nb::cast<bool>(result) == false) {
                throw std::runtime_error("progress callback requested cancellation");
            }
        };
    }
    {
        nb::gil_scoped_release release;
        const auto parsed = nlohmann::json::parse(requestText);
        responseText = vc::lasagna::rankSnapPairsJson(parsed, callback).dump();
    }
    return nb::cast<nb::dict>(json.attr("loads")(responseText));
}

} // namespace

NB_MODULE(vc_lasagna_amgx, m)
{
    m.doc() = "Lasagna AMGX screened-Laplace snap ranking bindings";
    m.def("rank_snap_pairs",
          &rankSnapPairs,
          "request"_a,
          "progress_callback"_a = nb::none());
}
