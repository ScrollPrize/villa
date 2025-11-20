#pragma once

#include <QObject>
#include <QDateTime>
#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <optional>

#include <nlohmann/json.hpp>
 

 
struct ColPoint
{
    uint64_t id;
    uint64_t collectionId;
    cv::Vec3f p = {0,0,0};
    float winding_annotation = NAN;
    qint64 creation_time = 0;
};
 
struct CollectionMetadata
{
    bool absolute_winding_number = false;
};
 
class VCCollection : public QObject
{
    Q_OBJECT

public:
    enum class WindingFillMode {
        Incremental,
        Decremental,
        Constant
    };

    struct Collection
    {
        uint64_t id;
        std::string name;
        std::unordered_map<uint64_t, ColPoint> points;
        CollectionMetadata metadata;
        cv::Vec3f color;
    };

    explicit VCCollection(QObject* parent = nullptr);
    ~VCCollection();

    uint64_t addCollection(const std::string& name);
    ColPoint addPoint(const std::string& collectionName, const cv::Vec3f& point);
    void addPoints(const std::string& collectionName, const std::vector<cv::Vec3f>& points);
    void updatePoint(const ColPoint& point);
    void removePoint(uint64_t pointId);

    void clearCollection(uint64_t collectionId);
    void clearAll();
    void renameCollection(uint64_t collectionId, const std::string& newName);

    uint64_t getCollectionId(const std::string& name) const;
    const std::unordered_map<uint64_t, Collection>& getAllCollections() const;
    void setCollectionMetadata(uint64_t collectionId, const CollectionMetadata& metadata);
    void setCollectionColor(uint64_t collectionId, const cv::Vec3f& color);
    std::optional<ColPoint> getPoint(uint64_t pointId) const;
    std::vector<ColPoint> getPoints(const std::string& collectionName) const;
    std::string generateNewCollectionName(const std::string& prefix = "col") const;
    void autoFillWindingNumbers(uint64_t collectionId, WindingFillMode mode);

   bool saveToJSON(const std::string& filename) const;
   bool loadFromJSON(const std::string& filename);

signals:
   void collectionChanged(uint64_t collectionId); // Generic signal for name/metadata changes
    void collectionsAdded(const std::vector<uint64_t>& collectionIds);
    void collectionRemoved(uint64_t collectionId);

    void pointAdded(const ColPoint& point);
    void pointChanged(const ColPoint& point);
    void pointRemoved(uint64_t pointId);


private:
    uint64_t getNextPointId();
    uint64_t getNextCollectionId();
    
    std::optional<uint64_t> findCollectionByName(const std::string& name) const;
    uint64_t findOrCreateCollectionByName(const std::string& name);

    std::unordered_map<uint64_t, Collection> _collections;
    std::unordered_map<uint64_t, ColPoint> _points;
    uint64_t _next_point_id = 1;
    uint64_t _next_collection_id = 1;
};
 
void to_json(nlohmann::json& j, const ColPoint& p);
void from_json(const nlohmann::json& j, ColPoint& p);
 
void to_json(nlohmann::json& j, const CollectionMetadata& m);
void from_json(const nlohmann::json& j, CollectionMetadata& m);
 
void to_json(nlohmann::json& j, const VCCollection::Collection& c);
void from_json(const nlohmann::json& j, VCCollection::Collection& c);
 
