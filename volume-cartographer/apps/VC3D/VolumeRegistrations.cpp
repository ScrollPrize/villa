#include "VolumeRegistrations.hpp"

#include <vc_object_store/Uuid.hpp>
#include <vc_object_store/objects/ObjectStore.hpp>
#include <nlohmann/json.hpp>

#include <regex>
#include <set>
#include <stdexcept>

namespace vc3d::registrations {

std::vector<Related> related(const vc::objects::VolumeTransforms& transforms, const std::string& volume)
{
    std::set<std::string> volumes;
    for (const auto& registration : transforms.graph.registrations()) {
        volumes.insert(registration.source);
        volumes.insert(registration.target);
    }
    volumes.erase(volume);
    std::vector<Related> result;
    for (const auto& other : volumes) {
        auto routes = transforms.graph.resolve(other, volume);
        if (routes.status == vcr::catalog::TransformStatus::Unavailable) continue;
        Related entry;
        entry.volume = other;
        entry.status = transforms.resolve(other, volume).status;
        entry.routes = std::move(routes.candidates);
        if (const auto chosen = transforms.choices.find({other, volume}); chosen != transforms.choices.end())
            entry.chosen = chosen->second;
        result.push_back(std::move(entry));
    }
    return result;
}

std::string describeRoute(const vcr::catalog::RegistrationGraph& graph, const std::vector<std::string>& path)
{
    std::string text;
    for (const auto& step : path) {
        constexpr std::string_view inversePrefix = "inverse:";
        const bool inverse = step.starts_with(inversePrefix);
        const auto key = inverse ? step.substr(inversePrefix.size()) : step;
        std::string label = key;
        for (const auto& registration : graph.registrations())
            if (registration.id + "@" + registration.revision == key) {
                label = (registration.provenance.empty() ? registration.id : registration.provenance) +
                        " (" + registration.source + " → " + registration.target + ")";
                break;
            }
        if (!text.empty()) text += ", then ";
        text += inverse ? "inverse of " + label : label;
    }
    return text;
}

vc::objects::VolumeAffine parseAffine(const std::string& text)
{
    static const std::regex number(R"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)");
    std::vector<double> values;
    for (auto it = std::sregex_iterator(text.begin(), text.end(), number); it != std::sregex_iterator(); ++it)
        values.push_back(std::stod(it->str()));
    if (values.size() != 12 && values.size() != 16)
        throw std::invalid_argument("expected three or four rows of four numbers, got " + std::to_string(values.size()) + " numbers");
    auto rows = nlohmann::json::array();
    for (size_t row = 0; row < values.size() / 4; ++row)
        rows.push_back({values[row * 4], values[row * 4 + 1], values[row * 4 + 2], values[row * 4 + 3]});
    // fromJson checks the homogeneous row and that the matrix inverts.
    return vc::objects::VolumeAffine::fromJson(rows);
}

std::string registerAffine(vc::objects::ObjectStoreSession& session, const vc::objects::AttachmentIdentity& attachment,
                           const std::string& source, const std::string& target,
                           const vc::objects::VolumeAffine& matrix, const std::string& name)
{
    if (source.empty() || target.empty()) throw std::invalid_argument("a registration needs a source and a target volume");
    if (source == target) throw std::invalid_argument("a volume cannot be registered to itself");
    const auto uuid = vc::objects::util::newUuid();
    session.act(attachment, {{uuid, std::nullopt}}, [&](auto& db, auto& txn) {
        db.registerVolumeTransform(txn, source, target, matrix, name.empty() ? source + " → " + target : name, uuid);
    });
    return uuid;
}

void chooseRoute(vc::objects::ObjectStoreSession& session, const vc::objects::AttachmentIdentity& attachment,
                 const std::string& source, const std::string& target, const std::vector<std::string>& path)
{
    // The pair's existing choice is updated in place; otherwise a new one is made.
    std::string uuid;
    std::optional<std::string> revision;
    session.read(attachment, [&](const auto& db, const auto& txn, const auto&) {
        for (const auto& record : db.objectsByKind(txn, "volume_transform_choice")) {
            const auto choice = nlohmann::json::parse(record.meta).value("choice", nlohmann::json::object());
            if (choice.value("source_volume", "") == source && choice.value("target_volume", "") == target) {
                uuid = record.uuid;
                revision = db.currentObjectRevision(txn, uuid)->uuid;
            }
        }
    });
    if (uuid.empty()) uuid = vc::objects::util::newUuid();
    session.act(attachment, {{uuid, revision}}, [&](auto& db, auto& txn) {
        db.chooseVolumeTransform(txn, source, target, path, uuid);
    });
}

} // namespace vc3d::registrations
