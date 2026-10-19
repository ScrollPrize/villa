#pragma once
#include "StorageBindings.hpp"
#include <nanobind/ndarray.h>
#include <thread>

namespace {
struct PyPackedGeometryArray {
    std::shared_ptr<vo::PackedGeometryArray> packed;
    nb::capsule owner() const {
        return nb::capsule(new std::shared_ptr<vo::PackedGeometryArray>(packed), [](void* p) noexcept {
            delete static_cast<std::shared_ptr<vo::PackedGeometryArray>*>(p);
        });
    }
    auto data() const {
        return nb::ndarray<const uint8_t, nb::numpy, nb::ndim<1>>(
            packed->data.data(), {packed->data.size()}, owner());
    }
};

struct PyGraphReadView {
    // Destruction order releases the transaction before its store.
    std::shared_ptr<vo::ObjectStore> store;
    std::unique_ptr<vo::ReadTxn> txn;
    std::shared_ptr<vo::GraphReadView> graph;
    std::thread::id thread = std::this_thread::get_id();

    void require() const {
        if (thread != std::this_thread::get_id()) throw std::runtime_error("graph read view is thread-confined");
        if (!txn) throw std::runtime_error("graph read view is closed");
    }
    void close() {
        if (thread != std::this_thread::get_id()) throw std::runtime_error("graph read view is thread-confined");
        txn.reset(); graph.reset(); store.reset();
    }
    nb::capsule owner() const {
        return nb::capsule(new std::shared_ptr<vo::GraphReadView>(graph), [](void* p) noexcept {
            delete static_cast<std::shared_ptr<vo::GraphReadView>*>(p);
        });
    }
    auto array(const std::vector<uint64_t>& values, size_t start, size_t count) const {
        require();
        // const ndarray makes views read-only; its capsule keeps only the
        // resident arrays alive, not the database transaction.
        const uint64_t* data = values.empty() ? nullptr : values.data() + start;
        return nb::ndarray<const uint64_t, nb::numpy, nb::ndim<1>>(data, {count}, owner());
    }
};

void bindGraphReadView(nb::module_& m) {
    nb::class_<PyPackedGeometryArray>(m, "PackedGeometryArray")
        .def_prop_ro("dtype", [](PyPackedGeometryArray& self) { return self.packed->dtype; })
        .def_prop_ro("shape", [](PyPackedGeometryArray& self) { return self.packed->shape; })
        .def_prop_ro("data", &PyPackedGeometryArray::data, nb::rv_policy::reference)
        .def_prop_ro("offsets", [](PyPackedGeometryArray& self) {
            return nb::ndarray<const uint64_t, nb::numpy, nb::ndim<1>>(
                self.packed->offsets.data(), {self.packed->offsets.size()}, self.owner());
        }, nb::rv_policy::reference)
        .def_prop_ro("values", [](PyPackedGeometryArray& self) {
            // NumPy views preserve stored byte order/dtype without conversion or
            // another payload copy. The ndarray capsule owns the native buffer.
            nb::object bytes = nb::cast(self.data(), nb::rv_policy::reference);
            return bytes.attr("view")(self.packed->dtype).attr("reshape")(nb::cast(self.packed->shape));
        });
    nb::class_<PyGraphReadView>(m, "GraphReadView")
        .def("__enter__", [](nb::object self) { nb::cast<PyGraphReadView&>(self).require(); return self; })
        .def("__exit__", [](PyGraphReadView& self, nb::args) { self.close(); })
        .def("close", &PyGraphReadView::close)
        .def_prop_ro("object_uuids", [](PyGraphReadView& self) { self.require(); return self.graph->objectUuids; })
        .def_prop_ro("edge_uuids", [](PyGraphReadView& self) { self.require(); return self.graph->edgeUuids; })
        .def_prop_ro("geometry_revisions", [](PyGraphReadView& self) { self.require(); return self.graph->geometryRevisions; })
        .def_prop_ro("generations", [](PyGraphReadView& self) { self.require(); return self.graph->generations; })
        .def_prop_ro("journal_sequence", [](PyGraphReadView& self) { self.require(); return self.graph->journalSequence; })
        .def_prop_ro("offsets", [](PyGraphReadView& self) {
            self.require(); return self.array(self.graph->offsets, 0, self.graph->offsets.size());
        }, nb::rv_policy::reference)
        .def_prop_ro("partners", [](PyGraphReadView& self) {
            self.require(); return self.array(self.graph->neighbours, 0, self.graph->neighbours.size());
        }, nb::rv_policy::reference)
        .def_prop_ro("edge_indices", [](PyGraphReadView& self) {
            self.require(); return self.array(self.graph->edgeIndices, 0, self.graph->edgeIndices.size());
        }, nb::rv_policy::reference)
        .def_prop_ro("endpoints", [](PyGraphReadView& self) {
            self.require();
            return nb::ndarray<const uint64_t, nb::numpy, nb::ndim<2>>(
                self.graph->endpoints.data(), {self.graph->endpoints.size(), size_t(2)}, self.owner());
        }, nb::rv_policy::reference)
        .def("neighbours", [](PyGraphReadView& self, size_t node) {
            self.require();
            if (node >= self.graph->objectUuids.size()) throw std::out_of_range("graph node out of range");
            const auto begin = self.graph->offsets[node], end = self.graph->offsets[node+1];
            return self.array(self.graph->neighbours, begin, end - begin);
        }, nb::arg("node"))
        .def("objects", [](PyGraphReadView& self, const std::optional<std::vector<std::string>>& fields) {
            self.require();
            std::vector<vo::ObjectRecord> records;
            if (!self.graph->objectUuids.empty()) {
                vo::ObjectSelection selection; selection.uuids = self.graph->objectUuids;
                nb::gil_scoped_release release;
                records = self.store->selectObjects(*self.txn, selection, fields);
            }
            std::map<std::string, size_t> positions;
            for (size_t i = 0; i < records.size(); ++i) positions.emplace(records[i].uuid, i);
            const std::set<std::string> projection = fields ? std::set<std::string>(fields->begin(), fields->end()) : std::set<std::string>{};
            nb::list out;
            for (const auto& id : self.graph->objectUuids) out.append(objectToPy(records.at(positions.at(id)), fields ? &projection : nullptr));
            return out;
        }, nb::arg("fields") = std::nullopt)
        .def("edges", [](PyGraphReadView& self, const std::optional<std::vector<std::string>>& fields) {
            self.require();
            vo::EdgeSelection selection; selection.uuids = self.graph->edgeUuids;
            std::vector<vo::EdgeRecord> records;
            { nb::gil_scoped_release release; records = self.store->selectEdges(*self.txn, selection, fields); }
            const std::set<std::string> projection = fields ? std::set<std::string>(fields->begin(), fields->end()) : std::set<std::string>{};
            nb::list out;
            for (const auto& record : records) out.append(edgeToPy(record, fields ? &projection : nullptr));
            return out;
        }, nb::arg("fields") = std::nullopt)
        .def("read_geometry_arrays", [](PyGraphReadView& self, const std::string& name,
                                        const std::optional<std::vector<size_t>>& nodes) {
            self.require();
            std::vector<std::string> revisions;
            auto append = [&](size_t node) {
                const auto& revision = self.graph->geometryRevisions.at(node);
                if (!revision) throw std::invalid_argument("graph node has no stored geometry");
                revisions.push_back(*revision);
            };
            if (nodes) for (auto node : *nodes) append(node);
            else for (size_t node = 0; node < self.graph->objectUuids.size(); ++node) append(node);
            std::shared_ptr<vo::PackedGeometryArray> packed;
            { nb::gil_scoped_release release;
              packed = std::make_shared<vo::PackedGeometryArray>(self.store->readGeometryArrays(*self.txn, revisions, name)); }
            return PyPackedGeometryArray{std::move(packed)};
        }, nb::arg("array_name"), nb::arg("nodes") = std::nullopt)
        .def("read_geometry_ranges", [](PyGraphReadView& self, size_t node, const std::string& name,
                                         const std::vector<std::array<int64_t, 2>>& ranges) {
            self.require();
            const auto& revision = self.graph->geometryRevisions.at(node);
            if (!revision) throw std::invalid_argument("graph node has no stored geometry");
            std::vector<std::vector<uint8_t>> values;
            { nb::gil_scoped_release release; values = self.store->readGeometryRanges(*self.txn, *revision, name, ranges); }
            nb::list out;
            for (const auto& bytes : values) out.append(nb::bytes(reinterpret_cast<const char*>(bytes.data()), bytes.size()));
            return out;
        }, nb::arg("node"), nb::arg("array_name"), nb::arg("ranges"));
}
}
