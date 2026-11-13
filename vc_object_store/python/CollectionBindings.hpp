#pragma once
#include "vc_object_store/objects/InitialImport.hpp"
#include "vc_object_store/objects/PackedStore.hpp"
#include "vc_object_store/objects/imports/ParallelImportStream.hpp"
#include "vc_object_store/objects/FiberConstraintGraph.hpp"
#include "vc_object_store/objects/VolumeTransforms.hpp"
#include "vc_object_store/geometry/ArrayBytes.hpp"
#include "FiberResultBindings.hpp"
#include "FiberGraphBindings.hpp"
#include "internal/Publication.hpp"
#include <nanobind/stl/unique_ptr.h>
#include <iostream>
#include <chrono>

namespace {
// One pinned composed-store snapshot; no connection is shared between processes.
struct PyCollectionReader {
    std::unique_ptr<vo::ObjectCollection> collection;
    std::unique_ptr<vo::CollectionTxn> txn;
    PyCollectionReader(const std::filesystem::path& path,const std::vector<std::filesystem::path>& files,
                       const std::vector<std::filesystem::path>& directories) {
        vo::OpenOptions options;options.readOnly=true;
        options.storeReferenceFiles=files;options.storeReferenceDirectories=directories;
        collection=std::make_unique<vo::ObjectCollection>(vo::ObjectCollection::open(path,options));
        txn=std::make_unique<vo::CollectionTxn>(collection->read());
    }
    void check() const {if(!txn)throw std::runtime_error("collection reader is closed");}
    void close(){txn.reset();collection.reset();}
};

std::vector<vo::InitialExtensionRow> initialExtensions(const nb::dict& row) {
    std::vector<vo::InitialExtensionRow> result;
    if(has(row,"extensions"))for(const auto input:nb::cast<nb::sequence>(row["extensions"])) {
        const auto value=nb::cast<nb::dict>(input);
        result.push_back({getStr(value,"table"),extensionRowFromPy(value["values"])});
    }
    return result;
}
}

inline void bindCollectionApi(nb::module_& module) {
    using namespace nb::literals;
    nb::class_<PyCollectionReader>(module,"CollectionReader",
        "Read one logical snapshot across plain, packed, sharded and referenced stores.")
    .def(nb::init<const std::filesystem::path&,const std::vector<std::filesystem::path>&,
                 const std::vector<std::filesystem::path>&>(),"path"_a,
         "store_reference_files"_a=std::vector<std::filesystem::path>{},
         "store_reference_directories"_a=std::vector<std::filesystem::path>{},
         nb::call_guard<nb::gil_scoped_release>())
    .def("close",&PyCollectionReader::close)
    .def("__enter__",[](PyCollectionReader& self)->PyCollectionReader&{self.check();return self;},nb::rv_policy::reference_internal)
    .def("__exit__",[](PyCollectionReader& self,nb::args){self.close();})
    .def("resolve_store_position",[](PyCollectionReader& self,const nb::dict& input){
        self.check();
        auto position=nlohmann::json::parse(jsonDumps(input));
        if(position.contains("store")) {
            if(position.contains("store_uuid"))throw std::invalid_argument("store position has two store keys");
            position["store_uuid"]=position.at("store");position.erase("store");
        }
        const auto source=vo::extensionSourcesFromJson(nlohmann::json::array({position}).dump()).at(0);
        nb::gil_scoped_release release;
        return vo::resolveStorePosition(*self.collection,*self.txn,source).string();
    },"position"_a,"Resolve an exact provisioned source snapshot to its public root locator.")
    .def_prop_ro("source",[](PyCollectionReader& self){
        self.check();const auto p=self.collection->provisioning(*self.txn);
        nb::dict result;result["store_uuid"]=self.collection->uuid(*self.txn);
        result["generation"]=p.generation;result["commit_sequence"]=p.commitSequence;return result;
    })
    .def_prop_ro("members",[](PyCollectionReader& self){
        self.check();nb::list result;
        for(const auto& member:self.collection->members()) {
            nb::dict row;row["database_uuid"]=member.databaseUuid;row["store_uuid"]=member.storeUuid;
            row["generation"]=member.generation;row["commit_sequence"]=member.commitSequence;
            row["locator"]=member.locator.string();row["partition"]=member.partition;
            row["container"]=vo::isPackedDatabasePath(member.locator)?vo::packedContainerPath(member.locator).string():member.locator.string();
            result.append(row);
        }
        return result;
    },"Resolved read-only physical members and their snapshot pins; no object directory is created.")
    .def("objects",[](PyCollectionReader& self,const std::optional<std::string>& kind){
        self.check();std::vector<vo::ObjectRecord> records;
        self.collection->scanObjects(*self.txn,[&](const auto& row){records.push_back(row.record);return true;},kind);
        std::sort(records.begin(),records.end(),[](const auto& a,const auto& b){return a.uuid<b.uuid;});
        nb::list result;for(const auto& row:records)result.append(recordToPy(row));return result;
    },"kind"_a=nb::none())
    .def("object",[](PyCollectionReader& self,const std::string& uuid)->nb::object{
        self.check();const auto row=self.collection->object(*self.txn,{"",uuid});
        return row?nb::object(recordToPy(row->record)):nb::object(nb::none());
    },"uuid"_a)
    .def("geometry_arrays",[](PyCollectionReader& self,const std::vector<std::string>& uuids,const std::string& name,
                              const std::optional<std::string>& volume){
        self.check();std::vector<std::vector<uint8_t>> bytes;std::vector<vo::GeometryRevision> descriptors;
        const auto transforms=volume?std::optional(self.collection->volumeTransforms(*self.txn)):std::nullopt;
        {for(const auto& uuid:uuids){
            auto geometry=self.collection->currentGeometry(*self.txn,{"",uuid});
            if(!geometry)throw std::invalid_argument("object has no geometry: "+uuid);
            bytes.push_back(self.collection->readGeometryArray(*self.txn,{"",uuid},name));
            if(volume){
                const auto object=self.collection->object(*self.txn,{"",uuid});
                const auto mapping=object?transforms->toVolume(nlohmann::json::parse(object->record.meta),*volume):std::nullopt;
                if(!mapping)throw std::invalid_argument("object has no unambiguous mapping to target volume: "+uuid);
                const auto array=std::find_if(geometry->arrays.begin(),geometry->arrays.end(),[&](const auto& a){return a.name==name;});
                if(array==geometry->arrays.end()||array->dtype!="<f8"||array->shape.size()!=2||array->shape[1]!=3)
                    throw std::invalid_argument("mapped geometry requires float64 XYZ points");
                auto points=vo::xyzFromBytes<double>(bytes.back());
                for(auto& point:points)point=mapping->apply(point);
                bytes.back()=vo::detail::xyzArray<double>(name,"<f8",array->shape,points).bytes;
            }
            descriptors.push_back(std::move(*geometry));
        }}
        nb::list result;for(size_t i=0;i<uuids.size();++i){
            const auto& arrays=descriptors[i].arrays;
            const auto found=std::find_if(arrays.begin(),arrays.end(),[&](const auto& a){return a.name==name;});
            if(found==arrays.end())throw std::invalid_argument("missing geometry array: "+name);
            nb::dict row;row["uuid"]=uuids[i];row["dtype"]=found->dtype;row["shape"]=found->shape;
            row["data"]=nb::bytes(reinterpret_cast<const char*>(bytes[i].data()),bytes[i].size());result.append(row);
        }return result;
    },"uuids"_a,"array"_a,"volume"_a=nb::none())
    .def("edges",[](PyCollectionReader& self,const std::string& kind){
        self.check();std::vector<vo::CollectionEdge> edges;
        edges=self.collection->edgesByKind(*self.txn,kind);
        nb::list result;for(const auto& edge:edges)result.append(recordToPy(edge.record));return result;
    },"kind"_a)
    .def("edge",[](PyCollectionReader& self,const std::string& uuid)->nb::object{
        self.check();const auto edge=self.collection->edge(*self.txn,{"",uuid});
        return edge?nb::object(recordToPy(edge->record)):nb::object(nb::none());
    },"uuid"_a)
    .def("fiber_graph_columns",[](PyCollectionReader& self,size_t workers,bool progress){
        self.check();vo::FiberGraphColumns columns;
        std::function<void(const char*,size_t,size_t)> report;
        const auto started=std::chrono::steady_clock::now();
        if(progress)report=[started](const char* stage,size_t rows,size_t partition){
            std::clog<<"{\"stage\":\"fiber_graph_"<<stage<<"\",\"rows\":"<<rows
                     <<",\"partition\":"<<partition<<",\"seconds\":"
                     <<std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()<<"}\n"<<std::flush;
        };
        {nb::gil_scoped_release release;columns=vo::readFiberGraphColumns(*self.collection,*self.txn,workers,report);}
        return fiberGraphColumnsToPy(std::move(columns),workers);
    },"workers"_a=32,"progress"_a=false,"One native parallel numeric graph snapshot; dense endpoints, binary edge UUIDs and selected node lengths.")
    .def("fiber_constraints",[](PyCollectionReader& self){
        self.check();std::vector<vo::CompactFiberConstraint> records;
        {nb::gil_scoped_release release;records=vo::readCompactFiberConstraintGraph(*self.collection,*self.txn);}
        std::vector<int64_t> ids;std::vector<double> values;ids.reserve(records.size());values.reserve(records.size()*16);
        nb::list uuids,a,b;
        for(const auto& r:records){ids.push_back(int64_t(r.runtimeIndex));uuids.append(vo::fiberGraphUuid(r.uuid));a.append(vo::fiberGraphUuid(r.a));b.append(vo::fiberGraphUuid(r.b));
            values.push_back(r.arcA);values.push_back(r.arcB);
            values.insert(values.end(),r.normalA.begin(),r.normalA.end());values.insert(values.end(),r.normalB.begin(),r.normalB.end());
            values.insert(values.end(),r.connector.begin(),r.connector.end());
            values.push_back(r.windingDistance);values.push_back(r.parallel);
            values.push_back(r.validA);values.push_back(r.validB);values.push_back(r.validConnector);}
        nb::dict result;result["edge_uuid"]=uuids;result["fiber_a"]=a;result["fiber_b"]=b;
        result["ids"]=nb::bytes(reinterpret_cast<const char*>(ids.data()),ids.size()*sizeof(int64_t));
        result["values"]=nb::bytes(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(double));
        return result;
    },"Bulk UUID-ordered compact graph buffers (host int64 runtime indices/float64 decoded values). No stored endpoint XYZ.")
    .def("extension_tables",[](PyCollectionReader& self){
        self.check();nb::list result;for(const auto& table:self.collection->extensionTables(*self.txn))result.append(extensionTableToPy(table));return result;
    })
    .def("fiber_result_columns",[](PyCollectionReader& self,const std::string& run){
        self.check();vo::FiberGraphResultColumns columns;
        {nb::gil_scoped_release release;columns=vo::readFiberGraphResultColumns(*self.collection,*self.txn,run);}
        return fiberResultColumnsToPy(std::move(columns));
    },"run_uuid"_a,"Owned UUID-ordered NumPy numeric columns for one run, with explicit NULL masks.")
    .def("extension_rows",[](PyCollectionReader& self,const std::string& name,const nb::sequence& key,const nb::dict& filter){
        self.check();const auto prefix=extensionRowFromPy(key);const auto selection=extensionKeyFilterFromPy(filter);nb::list result;
        self.collection->scanExtensionRows(*self.txn,name,[&](const auto& row){result.append(extensionRowToPy(row));return true;},prefix,selection);return result;
    },"table"_a,"key_prefix"_a=nb::tuple(),"key_filter"_a=nb::dict());

    // These records are already decoded/prepared. Native source imports use
    // the same InitialImport and ParallelImportStream, not another SQL writer.
    module.def("import_initial_records",[](const std::filesystem::path& destination,const nb::sequence& objects,
            const nb::sequence& edges,const nb::sequence& definitions,const std::vector<std::filesystem::path>& dependencies,
            size_t workers,bool packed,bool edgeAdjacencyIndex,
            const std::optional<std::vector<std::string>>& edgeColumns){
        if(!workers||workers>1024)throw std::invalid_argument("workers must be in 1..1024");
        std::vector<vo::InitialObjectRecord> records;records.reserve(nb::len(objects));
        for(const auto input:objects){const auto row=nb::cast<nb::dict>(input);vo::InitialObjectRecord record;
            record.record=objectFromPy(nb::cast<nb::dict>(row["record"]));
            if(has(row,"geometry"))record.geometry=geometryFromPy(nb::cast<nb::dict>(row["geometry"]));
            record.extensions=initialExtensions(row);records.push_back(std::move(record));}
        std::vector<vo::InitialCollectionEdge> relations;relations.reserve(nb::len(edges));
        for(const auto input:edges){const auto row=nb::cast<nb::dict>(input);vo::InitialCollectionEdge edge;
            edge.record=edgeFromPy(nb::cast<nb::dict>(row["record"]));edge.a={"",edge.record.aUuid};edge.b={"",edge.record.bUuid};
            edge.extensions=initialExtensions(row);relations.push_back(std::move(edge));}
        vo::InitialImportOptions options;options.shards.resize(workers);options.dependencies=dependencies;
        const auto stride=INT64_MAX/static_cast<int64_t>(workers);
        for(size_t i=0;i<workers;++i){const std::array<int64_t,2> range{
            1+static_cast<int64_t>(i)*stride,static_cast<int64_t>(i+1)*stride};
            options.shards[i].objectIds=range;}
        options.store.edgeAdjacencyIndex=edgeAdjacencyIndex;options.store.edgeColumns=edgeColumns;
        options.store.author="fiber-workflow";if(packed)options.packed=vo::PackedStoreOptions{};
        for(const auto input:definitions)options.extensions.push_back(extensionTableFromPy(nb::cast<nb::dict>(input)));
        vo::InitialImportResult result;
        {nb::gil_scoped_release release;
            vo::imports::ParallelImportOptions queues;queues.shards=workers;queues.readers=std::min(workers,size_t(8));
            vo::imports::ParallelImportStream stream(queues);
            const size_t batch=1024;
            stream.start((records.size()+batch-1)/batch,(relations.size()+batch-1)/batch,
                [&](size_t,size_t task,auto& emitter){for(size_t i=task*batch;i<std::min(records.size(),(task+1)*batch);++i)
                    emitter.emit(vo::imports::ImportObject{&records[i]},sizeof(vo::InitialObjectRecord),records[i].record.uuid);},
                [&](size_t,size_t task,auto& emitter){for(size_t i=task*batch;i<std::min(relations.size(),(task+1)*batch);++i){
                    const auto uuid=relations[i].record.uuid;emitter.emit(std::move(relations[i]),sizeof(vo::InitialCollectionEdge),uuid);}});
            options.beforePublish=[&]{stream.finish();};
            try{result=vo::importInitialCollection(destination,options,[&](size_t shard,auto& writer){
                std::vector<vo::imports::ImportObject> batch;
                while(stream.popObjects(shard,batch,[&]{writer.checkCancelled();})){
                    std::vector<const vo::InitialObjectRecord*> refs;for(const auto& row:batch)refs.push_back(&row.get());writer.appendObjectRefs(refs);}
            },[&](size_t shard,auto& writer){writer.importEdges([&](auto& batch){return stream.popEdges(shard,batch,[&]{writer.checkCancelled();});});});}
            catch(...){stream.fail(std::current_exception());throw;}
        }
        nb::dict out;out["path"]=result.path.string();out["store_uuid"]=result.storeUuid;
        out["ingestion_seconds"]=result.ingestionSeconds;out["persistence_seconds"]=result.persistenceSeconds;return out;
    },"destination"_a,"objects"_a,"edges"_a=nb::list(),"extensions"_a=nb::list(),
      "dependencies"_a=std::vector<std::filesystem::path>{},"workers"_a=1,"packed"_a=false,
      "edge_adjacency_index"_a=true,"edge_columns"_a=std::nullopt);

    module.def("write_extension_store",[](const std::filesystem::path& path,
            const std::vector<std::filesystem::path>& bases,const nb::sequence& definitions,const nb::dict& payload){
        vo::OpenOptions options;options.history=vo::HistoryMode::Off;options.spatialIndex=false;options.author="fiber-workflow";
        vo::detail::StagedDatabase staging(path,false);
        {
            auto store=vo::ObjectCollection::create(staging.path,bases,options);auto txn=store.write();
            for(const auto table:definitions)store.createExtensionTable(txn,extensionTableFromPy(nb::cast<nb::dict>(table)));
            for(const auto [name,input]:payload){const auto rows=extensionRowsFromPy(nb::cast<nb::sequence>(input));
                store.appendExtensionRows(txn,nb::cast<std::string>(name),rows);}
            store.seal(txn);txn.commit();store.checkpoint();
        }
        staging.publish();
    },"destination"_a,"dependencies"_a,"definitions"_a,"rows"_a);
}
