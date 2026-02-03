#include "vc/ui/VCCollection.hpp"
#include <QDebug>
#include <algorithm>
#include <vector>
#include <fstream>
#include <nlohmann/json.hpp>

NLOHMANN_JSON_NAMESPACE_BEGIN
template <>
struct adl_serializer<cv::Vec3f> {
    static void to_json(json& j, const cv::Vec3f& v) {
        j = {v[0], v[1], v[2]};
    }

    static void from_json(const json& j, cv::Vec3f& v) {
        j.at(0).get_to(v[0]);
        j.at(1).get_to(v[1]);
        j.at(2).get_to(v[2]);
    }
};

template <>
struct adl_serializer<cv::Vec2f> {
    static void to_json(json& j, const cv::Vec2f& v) {
        j = {v[0], v[1]};
    }

    static void from_json(const json& j, cv::Vec2f& v) {
        j.at(0).get_to(v[0]);
        j.at(1).get_to(v[1]);
    }
};
NLOHMANN_JSON_NAMESPACE_END
 
#define VC_POINTCOLLECTIONS_JSON_VERSION "1"



using json = nlohmann::json;
 
void to_json(json& j, const ColPoint& p) {
    j = json{
        {"p", p.p},
        {"creation_time", p.creation_time}
    };
    if (!std::isnan(p.winding_annotation)) {
        j["wind_a"] = p.winding_annotation;
    } else {
        j["wind_a"] = nullptr;
    }
}
 
void from_json(const json& j, ColPoint& p) {
    j.at("p").get_to(p.p);
    if (j.contains("wind_a") && !j.at("wind_a").is_null()) {
        j.at("wind_a").get_to(p.winding_annotation);
    } else {
        p.winding_annotation = std::nan("");
    }
    if (j.contains("creation_time")) {
        j.at("creation_time").get_to(p.creation_time);
    } else {
        p.creation_time = 0;
    }
}
 
void to_json(json& j, const CollectionMetadata& m) {
    j = json{
        {"winding_is_absolute", m.absolute_winding_number}
    };
}
 
void from_json(const json& j, CollectionMetadata& m) {
    j.at("winding_is_absolute").get_to(m.absolute_winding_number);
}
 
void to_json(json& j, const VCCollection::Collection& c) {
   json points_obj = json::object();
   for(const auto& pair : c.points) {
       points_obj[std::to_string(pair.first)] = pair.second;
   }

    j = json{
        {"name", c.name},
        {"points", points_obj},
        {"metadata", c.metadata},
        {"color", c.color}
    };

    if (c.anchor2d.has_value()) {
        j["anchor2d"] = c.anchor2d.value();
    }
}
 
void from_json(const json& j, VCCollection::Collection& c) {
    j.at("name").get_to(c.name);

    json points_obj = j.at("points");
    if (points_obj.is_object()) {
        for (auto& [id_str, point_json] : points_obj.items()) {
            uint64_t id = std::stoull(id_str);
            ColPoint p = point_json.get<ColPoint>();
            p.id = id;
            c.points[id] = p;
        }
    }

    j.at("metadata").get_to(c.metadata);
    j.at("color").get_to(c.color);

    if (j.contains("anchor2d") && !j.at("anchor2d").is_null()) {
        c.anchor2d = j.at("anchor2d").get<cv::Vec2f>();
    } else {
        c.anchor2d = std::nullopt;
    }
}
 
VCCollection::VCCollection(QObject* parent)
    : QObject(parent)
{
}

VCCollection::~VCCollection() = default;

uint64_t VCCollection::addCollection(const std::string& name)
{
    return findOrCreateCollectionByName(name);
}

ColPoint VCCollection::addPoint(const std::string& collectionName, const cv::Vec3f& point)
{
    uint64_t collection_id = findOrCreateCollectionByName(collectionName);
    
    ColPoint new_point;
    new_point.id = getNextPointId();
    new_point.collectionId = collection_id;
    new_point.p = point;
    new_point.creation_time = QDateTime::currentMSecsSinceEpoch();
    
    _collections[collection_id].points[new_point.id] = new_point;
    _points[new_point.id] = new_point;
    
    emit pointAdded(new_point);
    return new_point;
}

void VCCollection::addPoints(const std::string& collectionName, const std::vector<cv::Vec3f>& points)
{
    uint64_t collection_id = findOrCreateCollectionByName(collectionName);
    auto& collection_points = _collections[collection_id].points;

    for (const auto& p : points) {
        ColPoint new_point;
        new_point.id = getNextPointId();
        new_point.collectionId = collection_id;
        new_point.p = p;
        new_point.creation_time = QDateTime::currentMSecsSinceEpoch();
        collection_points[new_point.id] = new_point;
        _points[new_point.id] = new_point;
        emit pointAdded(new_point);
    }
}

void VCCollection::updatePoint(const ColPoint& point)
{
    if (_points.count(point.id)) {
        _points[point.id] = point;
        if (_collections.count(point.collectionId)) {
            _collections.at(point.collectionId).points[point.id] = point;
        }
        emit pointChanged(point);
    }
}

void VCCollection::removePoint(uint64_t pointId)
{
    if (_points.count(pointId)) {
        uint64_t collection_id = _points.at(pointId).collectionId;
        _points.erase(pointId);
        if (_collections.count(collection_id)) {
            _collections.at(collection_id).points.erase(pointId);
        }
        emit pointRemoved(pointId);
    }
}

void VCCollection::clearCollection(uint64_t collectionId)
{
    if (_collections.count(collectionId)) {
        auto& collection = _collections.at(collectionId);
        for (const auto& pair : collection.points) {
            _points.erase(pair.first);
            emit pointRemoved(pair.first);
        }
        _collections.erase(collectionId);
        emit collectionRemoved(collectionId);
    }
}

void VCCollection::clearAll()
{
    for (auto& point_pair : _points) {
        emit pointRemoved(point_pair.first);
    }
    _collections.clear();
    _points.clear();
    emit collectionRemoved(-1); // Sentinel for "all removed"
}

void VCCollection::renameCollection(uint64_t collectionId, const std::string& newName)
{
    if (_collections.count(collectionId)) {
        _collections.at(collectionId).name = newName;
        emit collectionChanged(collectionId);
    }
}

uint64_t VCCollection::getCollectionId(const std::string& name) const
{
    auto it = findCollectionByName(name);
    return it.has_value() ? it.value() : 0;
}

const std::unordered_map<uint64_t, VCCollection::Collection>& VCCollection::getAllCollections() const
{
    return _collections;
}

void VCCollection::setCollectionMetadata(uint64_t collectionId, const CollectionMetadata& metadata)
{
    if (_collections.count(collectionId)) {
        _collections.at(collectionId).metadata = metadata;
        emit collectionChanged(collectionId);
    }
}

void VCCollection::setCollectionColor(uint64_t collectionId, const cv::Vec3f& color)
{
    if (_collections.count(collectionId)) {
        _collections.at(collectionId).color = color;
        emit collectionChanged(collectionId);
    }
}

void VCCollection::setCollectionAnchor2d(uint64_t collectionId, const std::optional<cv::Vec2f>& anchor)
{
    if (_collections.count(collectionId)) {
        _collections.at(collectionId).anchor2d = anchor;
        emit collectionChanged(collectionId);
    }
}

std::optional<cv::Vec2f> VCCollection::getCollectionAnchor2d(uint64_t collectionId) const
{
    if (_collections.count(collectionId)) {
        return _collections.at(collectionId).anchor2d;
    }
    return std::nullopt;
}

std::optional<ColPoint> VCCollection::getPoint(uint64_t pointId) const
{
    if (_points.count(pointId)) {
        return _points.at(pointId);
    }
    return std::nullopt;
}

std::vector<ColPoint> VCCollection::getPoints(const std::string& collectionName) const
{
    std::vector<ColPoint> points;
    auto collection_id_opt = findCollectionByName(collectionName);
    if (collection_id_opt) {
        const auto& collection = _collections.at(*collection_id_opt);
        for (const auto& pair : collection.points) {
            points.push_back(pair.second);
        }
    }
    return points;
}

std::string VCCollection::generateNewCollectionName(const std::string& prefix) const
{
    int i = 1;
    std::string new_name;
    do {
        new_name = prefix + std::to_string(i++);
        bool name_exists = false;
        for(const auto& pair : _collections) {
            if (pair.second.name == new_name) {
                name_exists = true;
                break;
            }
        }
        if (!name_exists) break;
    } while (true);
    return new_name;
}

void VCCollection::autoFillWindingNumbers(uint64_t collectionId, WindingFillMode mode)
{
    if (_collections.count(collectionId)) {
        auto& collection = _collections.at(collectionId);
        
        std::vector<ColPoint*> points_to_sort;
        for(auto& pair : collection.points) {
            points_to_sort.push_back(&pair.second);
        }

        std::sort(points_to_sort.begin(), points_to_sort.end(),
            [](const ColPoint* a, const ColPoint* b) {
                return a->id < b->id;
            });

        float winding_counter;
        if (mode == WindingFillMode::Decremental) {
            winding_counter = static_cast<float>(points_to_sort.size());
        } else {
            winding_counter = 1.0f;
        }

        for(ColPoint* point : points_to_sort) {
            switch (mode) {
                case WindingFillMode::Incremental:
                    point->winding_annotation = winding_counter;
                    winding_counter += 1.0f;
                    break;
                case WindingFillMode::Decremental:
                    point->winding_annotation = winding_counter;
                    winding_counter -= 1.0f;
                    break;
                case WindingFillMode::Constant:
                    point->winding_annotation = 0.0f;
                    break;
            }
            updatePoint(*point);
        }
    }
}
 
bool VCCollection::saveToJSON(const std::string& filename) const
{
    json j;
   j["vc_pointcollections_json_version"] = VC_POINTCOLLECTIONS_JSON_VERSION;
    json collections_obj = json::object();
    for(const auto& pair : _collections) {
        collections_obj[std::to_string(pair.first)] = pair.second;
    }
    j["collections"] = collections_obj;
 
    std::ofstream o(filename);
    if (!o.is_open()) {
        qWarning() << "Failed to open file for writing: " << QString::fromStdString(filename);
        return false;
    }
    o << j.dump(4);
    o.close();
    return true;
}
 
bool VCCollection::loadFromJSON(const std::string& filename)
{
    std::ifstream i(filename);
    if (!i.is_open()) {
        qWarning() << "Failed to open file for reading: " << QString::fromStdString(filename);
        return false;
    }

    json j;
    try {
        i >> j;
    } catch (json::parse_error& e) {
        qWarning() << "Failed to parse JSON: " << e.what();
        return false;
    }

    clearAll();

    try {
       if (!j.contains("vc_pointcollections_json_version") || j.at("vc_pointcollections_json_version").get<std::string>() != VC_POINTCOLLECTIONS_JSON_VERSION) {
           throw std::runtime_error("JSON file has incorrect version or is missing version info.");
       }

        json collections_obj = j.at("collections");
        if (!collections_obj.is_object()) {
            return false;
        }

        for (auto& [id_str, col_json] : collections_obj.items()) {
           uint64_t id = std::stoull(id_str);
            Collection col = col_json.get<Collection>();
            col.id = id;
            _collections[col.id] = col;
            for (auto& point_pair : _collections.at(col.id).points) {
                point_pair.second.collectionId = col.id;
            }
        }

    } catch (json::exception& e) {
        qWarning() << "Failed to extract data from JSON: " << e.what();
        return false;
    }

    // Recalculate next IDs
    _next_collection_id = 1;
    _next_point_id = 1;
    for (const auto& col_pair : _collections) {
        if (col_pair.first >= _next_collection_id) {
            _next_collection_id = col_pair.first + 1;
        }
        for (const auto& point_pair : col_pair.second.points) {
            if (point_pair.first >= _next_point_id) {
                _next_point_id = point_pair.first + 1;
            }
        }
    }

    // Rebuild the _points map
    _points.clear();
    for (const auto& col_pair : _collections) {
        for (const auto& point_pair : col_pair.second.points) {
            _points[point_pair.first] = point_pair.second;
        }
    }

    std::vector<uint64_t> collectionIds;
    for (const auto& [col_id, _] : _collections) {
        collectionIds.push_back(col_id);
    }
    emit collectionsAdded(collectionIds);

    return true;
}

bool VCCollection::saveToSegmentPath(const std::filesystem::path& segmentPath) const
{
    if (segmentPath.empty()) {
        return false;
    }

    auto filePath = segmentPath / "corrections.json";

    // Filter to only collections with anchor2d set
    json collections_obj = json::object();
    int anchoredCount = 0;
    for (const auto& pair : _collections) {
        if (pair.second.anchor2d.has_value()) {
            collections_obj[std::to_string(pair.first)] = pair.second;
            anchoredCount++;
        }
    }

    // If no anchored collections exist, remove the file if it exists
    if (anchoredCount == 0) {
        std::error_code ec;
        if (std::filesystem::exists(filePath, ec)) {
            std::filesystem::remove(filePath, ec);
            qInfo() << "Removed empty corrections file:" << QString::fromStdString(filePath.string());
        }
        return true;
    }

    json j;
    j["vc_pointcollections_json_version"] = VC_POINTCOLLECTIONS_JSON_VERSION;
    j["collections"] = collections_obj;

    std::ofstream o(filePath);
    if (!o.is_open()) {
        qWarning() << "Failed to open corrections file for writing:" << QString::fromStdString(filePath.string());
        return false;
    }
    o << j.dump(4);
    o.close();

    qInfo() << "Saved" << anchoredCount << "anchored correction collections to" << QString::fromStdString(filePath.string());
    return true;
}

void VCCollection::applyAnchorOffset(float offsetX, float offsetY)
{
    if (offsetX == 0.0f && offsetY == 0.0f) {
        return;
    }

    int updatedCount = 0;
    for (auto& pair : _collections) {
        if (pair.second.anchor2d.has_value()) {
            cv::Vec2f& anchor = pair.second.anchor2d.value();
            anchor[0] += offsetX;
            anchor[1] += offsetY;
            updatedCount++;
            emit collectionChanged(pair.first);
        }
    }

    if (updatedCount > 0) {
        qInfo() << "Applied offset (" << offsetX << "," << offsetY << ") to" << updatedCount << "correction anchors";
    }
}

bool VCCollection::loadFromSegmentPath(const std::filesystem::path& segmentPath)
{
    // Clear existing anchored collections (segment-specific corrections) before loading
    // This ensures switching segments replaces old corrections with new ones
    std::vector<uint64_t> toRemove;
    for (const auto& [id, col] : _collections) {
        if (col.anchor2d.has_value()) {
            toRemove.push_back(id);
        }
    }
    if (!toRemove.empty()) {
        qInfo() << "Clearing" << toRemove.size() << "existing anchored correction collections";
    }
    for (uint64_t id : toRemove) {
        clearCollection(id);
    }

    // If no path provided, just clear (already done above) and return success
    if (segmentPath.empty()) {
        return true;
    }

    auto filePath = segmentPath / "corrections.json";

    std::error_code ec;
    if (!std::filesystem::exists(filePath, ec)) {
        // No corrections file - this is normal, not an error
        qInfo() << "No corrections file at" << QString::fromStdString(filePath.string());
        return true;
    }

    std::ifstream i(filePath);
    if (!i.is_open()) {
        qWarning() << "Failed to open corrections file for reading:" << QString::fromStdString(filePath.string());
        return false;
    }

    json j;
    try {
        i >> j;
    } catch (json::parse_error& e) {
        qWarning() << "Failed to parse corrections JSON:" << e.what();
        return false;
    }

    try {
        if (!j.contains("vc_pointcollections_json_version") ||
            j.at("vc_pointcollections_json_version").get<std::string>() != VC_POINTCOLLECTIONS_JSON_VERSION) {
            qWarning() << "Corrections file has incorrect version";
            return false;
        }

        json collections_obj = j.at("collections");
        if (!collections_obj.is_object()) {
            return false;
        }

        std::vector<uint64_t> addedCollectionIds;
        int loadedCount = 0;

        for (auto& [id_str, col_json] : collections_obj.items()) {
            Collection col = col_json.get<Collection>();
            // Only load collections with anchor2d set
            if (!col.anchor2d.has_value()) {
                continue;
            }

            uint64_t id = std::stoull(id_str);
            col.id = id;
            _collections[col.id] = col;
            for (auto& point_pair : _collections.at(col.id).points) {
                point_pair.second.collectionId = col.id;
                _points[point_pair.first] = point_pair.second;
            }

            // Update next IDs
            if (col.id >= _next_collection_id) {
                _next_collection_id = col.id + 1;
            }
            for (const auto& point_pair : col.points) {
                if (point_pair.first >= _next_point_id) {
                    _next_point_id = point_pair.first + 1;
                }
            }

            addedCollectionIds.push_back(col.id);
            loadedCount++;
        }

        if (!addedCollectionIds.empty()) {
            emit collectionsAdded(addedCollectionIds);
        }

        qInfo() << "Loaded" << loadedCount << "anchored correction collections from" << QString::fromStdString(filePath.string());

    } catch (json::exception& e) {
        qWarning() << "Failed to extract data from corrections JSON:" << e.what();
        return false;
    }

    return true;
}
 
uint64_t VCCollection::getNextPointId()
{
    return _next_point_id++;
}

uint64_t VCCollection::getNextCollectionId()
{
    return _next_collection_id++;
}

std::optional<uint64_t> VCCollection::findCollectionByName(const std::string& name) const
{
    for (const auto& pair : _collections) {
        if (pair.second.name == name) {
            return pair.first;
        }
    }
    return std::nullopt;
}

uint64_t VCCollection::findOrCreateCollectionByName(const std::string& name)
{
    auto existing_id = findCollectionByName(name);
    if (existing_id) {
        return *existing_id;
    }

    uint64_t new_id = getNextCollectionId();
    cv::Vec3f color = {
        (float)rand() / RAND_MAX,
        (float)rand() / RAND_MAX,
        (float)rand() / RAND_MAX
    };
    _collections[new_id] = {new_id, name, {}, {}, color};
    emit collectionsAdded({new_id});
    return new_id;
}


