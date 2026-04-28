#include "vc/ui/VCCollection.hpp"
#include <QDebug>
#include <algorithm>
#include <vector>
#include <fstream>

using Json = utils::Json;

#define VC_POINTCOLLECTIONS_JSON_VERSION "1"

// -- Vec3f/Vec2f helpers --
static Json vec3f_to_json(const cv::Vec3f& v) {
    auto a = Json::array();
    a.push_back(Json((double)v[0]));
    a.push_back(Json((double)v[1]));
    a.push_back(Json((double)v[2]));
    return a;
}

static cv::Vec3f vec3f_from_json(const Json& j) {
    return {j.at(0).get_float(), j.at(1).get_float(), j.at(2).get_float()};
}

static Json vec2f_to_json(const cv::Vec2f& v) {
    auto a = Json::array();
    a.push_back(Json((double)v[0]));
    a.push_back(Json((double)v[1]));
    return a;
}

static cv::Vec2f vec2f_from_json(const Json& j) {
    return {j.at(0).get_float(), j.at(1).get_float()};
}

// -- to_json / from_json --
void to_json(Json& j, const ColPoint& p) {
    j = Json{
        {"p", vec3f_to_json(p.p)},
        {"creation_time", Json((int64_t)p.creation_time)}
    };
    if (!std::isnan(p.winding_annotation)) {
        j["wind_a"] = (double)p.winding_annotation;
    } else {
        j["wind_a"] = nullptr;
    }
}

void from_json(const Json& j, ColPoint& p) {
    p.p = vec3f_from_json(j.at("p"));
    if (j.contains("wind_a") && !j.at("wind_a").is_null()) {
        p.winding_annotation = j.at("wind_a").get_float();
    } else {
        p.winding_annotation = std::nan("");
    }
    if (j.contains("creation_time")) {
        p.creation_time = j.at("creation_time").get_int64();
    } else {
        p.creation_time = 0;
    }
}

void to_json(Json& j, const CollectionMetadata& m) {
    j = Json{
        {"winding_is_absolute", Json(m.absolute_winding_number)}
    };
}

void from_json(const Json& j, CollectionMetadata& m) {
    m.absolute_winding_number = j.at("winding_is_absolute").get_bool();
}

void to_json(Json& j, const VCCollection::Collection& c) {
    Json points_obj = Json::object();
    for (const auto& pair : c.points) {
        Json pj;
        to_json(pj, pair.second);
        points_obj[std::to_string(pair.first)] = pj;
    }

    Json metadata_j;
    to_json(metadata_j, c.metadata);

    j = Json{
        {"name", Json(c.name)},
        {"points", std::move(points_obj)},
        {"metadata", std::move(metadata_j)},
        {"color", vec3f_to_json(c.color)}
    };

    if (c.anchor2d.has_value()) {
        j["anchor2d"] = vec2f_to_json(c.anchor2d.value());
    }
    if (c.autoFillMode != VCCollection::WindingFillMode::None) {
        j["autoFillMode"] = Json(static_cast<int>(c.autoFillMode));
        j["autoFillConstant"] = Json((double)c.autoFillConstant);
    }
}

void from_json(const Json& j, VCCollection::Collection& c) {
    c.name = j.at("name").get_string();

    Json points_obj = j.at("points");  // copy — ref into at() cache gets evicted by nested at() calls
    if (points_obj.is_object()) {
        for (auto it = points_obj.begin(); it != points_obj.end(); ++it) {
            uint64_t id = std::stoull(it.key());
            ColPoint p;
            from_json(*it, p);
            p.id = id;
            c.points[id] = p;
        }
    }

    from_json(j.at("metadata"), c.metadata);
    c.color = vec3f_from_json(j.at("color"));

    if (j.contains("anchor2d") && !j.at("anchor2d").is_null()) {
        c.anchor2d = vec2f_from_json(j.at("anchor2d"));
    } else {
        c.anchor2d = std::nullopt;
    }
    if (j.contains("autoFillMode") && j.at("autoFillMode").is_number()) {
        int modeInt = j.at("autoFillMode").get_int();
        if (modeInt >= 0 && modeInt <= 3) {
            c.autoFillMode = static_cast<VCCollection::WindingFillMode>(modeInt);
        }
    }
    if (j.contains("autoFillConstant") && j.at("autoFillConstant").is_number()) {
        c.autoFillConstant = j.at("autoFillConstant").get_float();
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
    new_point.creation_time = static_cast<int64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
    
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
        new_point.creation_time = static_cast<int64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
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

void VCCollection::autoFillWindingNumbers(uint64_t collectionId, WindingFillMode mode, float constantValue)
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
                case WindingFillMode::None:
                    break;
                case WindingFillMode::Incremental:
                    point->winding_annotation = winding_counter;
                    winding_counter += 1.0f;
                    break;
                case WindingFillMode::Decremental:
                    point->winding_annotation = winding_counter;
                    winding_counter -= 1.0f;
                    break;
                case WindingFillMode::Constant:
                    point->winding_annotation = constantValue;
                    break;
            }
            updatePoint(*point);
        }
    }
}

void VCCollection::setAutoFillMode(uint64_t collectionId, WindingFillMode mode, float constantValue)
{
    if (_collections.count(collectionId)) {
        _collections.at(collectionId).autoFillMode = mode;
        _collections.at(collectionId).autoFillConstant = constantValue;
        emit collectionChanged(collectionId);
    }
}

VCCollection::WindingFillMode VCCollection::getAutoFillMode(uint64_t collectionId) const
{
    if (_collections.count(collectionId)) {
        return _collections.at(collectionId).autoFillMode;
    }
    return WindingFillMode::None;
}

float VCCollection::getAutoFillConstant(uint64_t collectionId) const
{
    if (_collections.count(collectionId)) {
        return _collections.at(collectionId).autoFillConstant;
    }
    return 0.0f;
}

float VCCollection::computeAutoFillValue(uint64_t collectionId) const
{
    if (!_collections.count(collectionId)) {
        return NAN;
    }
    const auto& col = _collections.at(collectionId);

    switch (col.autoFillMode) {
        case WindingFillMode::None:
            return NAN;
        case WindingFillMode::Constant:
            return col.autoFillConstant;
        case WindingFillMode::Incremental: {
            float maxVal = 0.0f;
            bool found = false;
            for (const auto& [pid, pt] : col.points) {
                if (!std::isnan(pt.winding_annotation)) {
                    if (!found || pt.winding_annotation > maxVal) {
                        maxVal = pt.winding_annotation;
                        found = true;
                    }
                }
            }
            return found ? maxVal + 1.0f : 1.0f;
        }
        case WindingFillMode::Decremental: {
            float minVal = 0.0f;
            bool found = false;
            for (const auto& [pid, pt] : col.points) {
                if (!std::isnan(pt.winding_annotation)) {
                    if (!found || pt.winding_annotation < minVal) {
                        minVal = pt.winding_annotation;
                        found = true;
                    }
                }
            }
            return found ? minVal - 1.0f : -1.0f;
        }
    }
    return NAN;
}
 
bool VCCollection::saveToJSON(const std::string& filename) const
{
    Json j;
    j["vc_pointcollections_json_version"] = VC_POINTCOLLECTIONS_JSON_VERSION;
    Json collections_obj = Json::object();
    for (const auto& pair : _collections) {
        Json cj;
        to_json(cj, pair.second);
        collections_obj[std::to_string(pair.first)] = cj;
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
    Json j;
    try {
        j = Json::parse_file(filename);
    } catch (const std::exception& e) {
        qWarning() << "Failed to parse JSON: " << e.what();
        return false;
    }

    clearAll();

    try {
        if (!j.contains("vc_pointcollections_json_version") || j.at("vc_pointcollections_json_version").get_string() != VC_POINTCOLLECTIONS_JSON_VERSION) {
            throw std::runtime_error("JSON file has incorrect version or is missing version info.");
        }

        Json collections_obj = j.at("collections");  // copy — ref into at() cache gets evicted by nested at() calls
        if (!collections_obj.is_object()) {
            return false;
        }

        for (auto it = collections_obj.begin(); it != collections_obj.end(); ++it) {
            uint64_t id = std::stoull(it.key());
            Collection col;
            from_json(*it, col);
            col.id = id;
            _collections[col.id] = col;
            for (auto& point_pair : _collections.at(col.id).points) {
                point_pair.second.collectionId = col.id;
            }
        }

    } catch (const std::exception& e) {
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
    Json collections_obj = Json::object();
    int anchoredCount = 0;
    for (const auto& pair : _collections) {
        if (pair.second.anchor2d.has_value()) {
            Json cj;
            to_json(cj, pair.second);
            collections_obj[std::to_string(pair.first)] = cj;
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

    Json j;
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

    Json j;
    try {
        j = Json::parse_file(filePath);
    } catch (const std::exception& e) {
        qWarning() << "Failed to parse corrections JSON:" << e.what();
        return false;
    }

    try {
        if (!j.contains("vc_pointcollections_json_version") ||
            j.at("vc_pointcollections_json_version").get_string() != VC_POINTCOLLECTIONS_JSON_VERSION) {
            qWarning() << "Corrections file has incorrect version";
            return false;
        }

        Json collections_obj = j.at("collections");  // copy — ref into at() cache gets evicted by nested at() calls
        if (!collections_obj.is_object()) {
            return false;
        }

        std::vector<uint64_t> addedCollectionIds;
        int loadedCount = 0;

        for (auto it = collections_obj.begin(); it != collections_obj.end(); ++it) {
            Collection col;
            from_json(*it, col);
            // Only load collections with anchor2d set
            if (!col.anchor2d.has_value()) {
                continue;
            }

            uint64_t id = std::stoull(it.key());
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

    } catch (const std::exception& e) {
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
        static_cast<float>(rand()) / static_cast<float>(RAND_MAX),
        static_cast<float>(rand()) / static_cast<float>(RAND_MAX),
        static_cast<float>(rand()) / static_cast<float>(RAND_MAX)
    };
    _collections[new_id] = {new_id, name, {}, {}, color};
    emit collectionsAdded({new_id});
    return new_id;
}


