#include "internal/JsonFields.hpp"

namespace vc::objects::detail {
nlohmann::json objectJson(const nlohmann::json& value)
{
    if(value.is_null()||(value.is_string()&&value.get_ref<const std::string&>().empty()))
        return nlohmann::json::object();
    if(!value.is_object())throw std::invalid_argument("object store: expected a JSON object or empty value");
    return value;
}
nlohmann::json parseObjectJson(const std::string& text)
{
    return text.empty()?nlohmann::json::object():objectJson(nlohmann::json::parse(text));
}
std::string canonicalObjectJson(const std::string& text)
{
    if(text.empty()||text=="{}"||text=="null"||text=="\"\"")return "{}";
    return parseObjectJson(text).dump();
}
const nlohmann::json& jsonArray(const nlohmann::json& value, const char* context)
{
    if (!value.is_array()) throw std::invalid_argument(std::string(context) + " must be an array");
    return value;
}

const nlohmann::json& jsonArrayField(const nlohmann::json& record, const char* key, bool optional)
{
    static const auto empty = nlohmann::json::array();
    if (!record.is_object()) throw std::invalid_argument(std::string("record containing ") + key + " must be an object");
    return optional && !record.contains(key) ? empty : jsonArray(record.at(key), key);
}

std::optional<std::array<double, 6>> bboxFromJson(const nlohmann::json& j)
{
    if (!j.is_array() || j.size() != 6) return std::nullopt;
    std::array<double, 6> box{};
    for (int i = 0; i < 6; ++i) box[size_t(i)] = j.at(size_t(i)).get<double>();
    return box;
}

std::array<double, 2> scaleFromJson(const nlohmann::json& value)
{
    std::array<double, 2> scale{};
    if (value.is_number()) {
        scale.fill(value.get<double>());
    } else if (value.is_array() && value.size() == 2) {
        scale = {value.at(0).get<double>(), value.at(1).get<double>()};
    } else {
        throw std::runtime_error("object store: surface scale must be a number or a two-element array");
    }
    for (double axis : scale)
        if (!std::isfinite(axis) || axis < 0)
            throw std::runtime_error("object store: surface scale must be finite and nonnegative");
    return scale;
}
} // namespace vc::objects::detail
