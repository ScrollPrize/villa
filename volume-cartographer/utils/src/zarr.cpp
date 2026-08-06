#include "utils/zarr.hpp"

#include <cmath>

#if !defined(_WIN32)
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

namespace utils {

// ---------------------------------------------------------------------------
// JSON wrapper functions
// ---------------------------------------------------------------------------

JsonValue json_parse(std::string_view text) {
    return Json::parse(text);
}

std::string json_serialize(const JsonValue& v, int indent) {
    return v.dump(indent);
}

JsonValue json_object(
    std::initializer_list<std::pair<const std::string, JsonValue>> pairs) {
    return JsonValue(pairs);
}

JsonValue json_array(std::initializer_list<JsonValue> values) {
    auto arr = Json::array();
    for (const auto& v : values) arr.push_back(v);
    return arr;
}

const JsonValue* json_find(const JsonValue& obj, const std::string& key) {
    if (!obj.is_object()) return nullptr;
    if (!obj.contains(key)) return nullptr;
    return &obj.at(key);
}

// ---------------------------------------------------------------------------
// detail namespace
// ---------------------------------------------------------------------------

namespace detail {

ZarrMetadata parse_zarray(std::string_view json_str) {
    auto root = json_parse(json_str);
    if (!root.is_object())
        throw std::runtime_error("zarr: .zarray root must be a JSON object");

    ZarrMetadata meta;
    meta.version = ZarrVersion::v2;

    // shape
    if (auto* p = json_find(root, "shape"); p && p->is_array())
        for (const auto& v : (*p))
            meta.shape.push_back(v.get_size_t());

    // chunks
    if (auto* p = json_find(root, "chunks"); p && p->is_array())
        for (const auto& v : (*p))
            meta.chunks.push_back(v.get_size_t());

    // dtype (e.g. "<u2")
    if (auto* p = json_find(root, "dtype"); p && p->is_string()) {
        const auto& ds = p->get_string();
        if (!ds.empty() && (ds[0] == '<' || ds[0] == '>' || ds[0] == '|'))
            meta.byte_order = ds[0];
        auto dt = parse_dtype(ds);
        if (!dt) throw std::runtime_error("zarr: unsupported dtype: " + std::string(ds));
        meta.dtype = *dt;
    }

    // compressor
    if (auto* p = json_find(root, "compressor"); p) {
        if (p->is_object()) {
            if (auto* cid = json_find(*p, "id"); cid && cid->is_string())
                meta.compressor_id = cid->get_string();
            // numcodecs uses "clevel" for blosc but "level" for zstd/gzip/zlib.
            if (auto* cl = json_find(*p, "clevel"); cl && cl->is_number())
                meta.compression_level = cl->get_int();
            else if (auto* lv = json_find(*p, "level"); lv && lv->is_number())
                meta.compression_level = lv->get_int();
        }
    }

    // fill_value
    if (auto* p = json_find(root, "fill_value"); p) {
        if (p->is_number())
            meta.fill_value = p->get_double();
        else if (p->is_null())
            meta.fill_value = std::nullopt;
    }

    // dimension_separator
    if (auto* p = json_find(root, "dimension_separator"); p && p->is_string())
        meta.dimension_separator = p->get_string();

    // filters: not parsed (delta, fixedscaleoffset, quantize are unused)

    return meta;
}

ZarrCodecConfig parse_codec_config(const JsonValue& jv) {
    ZarrCodecConfig cc;
    if (auto* n = json_find(jv, "name"); n && n->is_string())
        cc.name = n->get_string();
    if (auto* c = json_find(jv, "configuration"); c)
        cc.configuration = std::make_shared<JsonValue>(*c);
    return cc;
}

ZarrMetadata parse_zarr_json(std::string_view json_str) {
    auto root = json_parse(json_str);
    if (!root.is_object())
        throw std::runtime_error("zarr: zarr.json root must be a JSON object");

    ZarrMetadata meta;
    meta.version = ZarrVersion::v3;

    // node_type
    if (auto* p = json_find(root, "node_type"); p && p->is_string())
        meta.node_type = p->get_string();

    // shape
    if (auto* p = json_find(root, "shape"); p && p->is_array())
        for (const auto& v : (*p))
            meta.shape.push_back(v.get_size_t());

    // data_type
    if (auto* p = json_find(root, "data_type"); p && p->is_string()) {
        auto dt = parse_dtype_v3(p->get_string());
        if (!dt) throw std::runtime_error("zarr: unsupported v3 data_type: " + p->get_string());
        meta.dtype = *dt;
    }

    // chunk_grid
    if (auto* p = json_find(root, "chunk_grid"); p && p->is_object()) {
        if (auto* cfg = json_find(*p, "configuration"); cfg && cfg->is_object()) {
            if (auto* cs = json_find(*cfg, "chunk_shape"); cs && cs->is_array())
                for (const auto& v : (*cs))
                    meta.chunks.push_back(v.get_size_t());
        }
    }

    // chunk_key_encoding
    if (auto* p = json_find(root, "chunk_key_encoding"); p && p->is_object()) {
        if (auto* nm = json_find(*p, "name"); nm && nm->is_string())
            meta.chunk_key_encoding = nm->get_string();
        if (auto* cfg = json_find(*p, "configuration"); cfg && cfg->is_object()) {
            if (auto* sep = json_find(*cfg, "separator"); sep && sep->is_string())
                meta.dimension_separator = sep->get_string();
        }
    }
    // Default separators.
    if (meta.chunk_key_encoding == "default" && meta.dimension_separator == ".")
        meta.dimension_separator = "/";

    // fill_value
    if (auto* p = json_find(root, "fill_value"); p) {
        if (p->is_number())
            meta.fill_value = p->get_double();
        else if (p->is_null())
            meta.fill_value = std::nullopt;
    }

    // codecs
    if (auto* p = json_find(root, "codecs"); p && p->is_array()) {
        for (const auto& cv : (*p)) {
            if (!cv.is_object()) continue;
            auto cc = parse_codec_config(cv);

            // Detect sharding_indexed codec.
            if (cc.name == "sharding_indexed" && cc.configuration && cc.configuration->is_object()) {
                ShardConfig sc;
                const auto& cfg = *cc.configuration;
                if (auto* cs = json_find(cfg, "chunk_shape"); cs && cs->is_array())
                    for (const auto& v : (*cs))
                        sc.sub_chunks.push_back(v.get_size_t());
                // index_location is always "start" — ignore any "end" values
                if (auto* ic = json_find(cfg, "index_codecs"); ic && ic->is_array())
                    for (const auto& icv : (*ic))
                        if (icv.is_object()) sc.index_codecs.push_back(parse_codec_config(icv));
                if (auto* sc_codecs = json_find(cfg, "codecs"); sc_codecs && sc_codecs->is_array())
                    for (const auto& scv : (*sc_codecs))
                        if (scv.is_object()) sc.sub_codecs.push_back(parse_codec_config(scv));
                meta.shard_config = std::move(sc);
            }

            meta.codecs.push_back(std::move(cc));
        }
    }

    return meta;
}

JsonValue codec_config_to_json(const ZarrCodecConfig& cc) {
    JsonObject obj;
    obj["name"] = JsonValue(cc.name);
    if (cc.configuration && !cc.configuration->is_null())
        obj["configuration"] = *cc.configuration;
    return JsonValue(std::move(obj));
}

std::string serialize_zarr_json(const ZarrMetadata& meta) {
    JsonObject root;
    root["zarr_format"] = JsonValue(3);
    root["node_type"] = JsonValue(meta.node_type);

    // data_type
    root["data_type"] = JsonValue(std::string(dtype_string_v3(meta.dtype)));

    // shape
    {
        JsonArray arr;
        for (auto s : meta.shape) arr.push_back(JsonValue(s));
        root["shape"] = JsonValue(std::move(arr));
    }

    // chunk_grid
    {
        JsonObject cg;
        cg["name"] = JsonValue("regular");
        JsonObject cg_cfg;
        JsonArray cs;
        for (auto c : meta.chunks) cs.push_back(JsonValue(c));
        cg_cfg["chunk_shape"] = JsonValue(std::move(cs));
        cg["configuration"] = JsonValue(std::move(cg_cfg));
        root["chunk_grid"] = JsonValue(std::move(cg));
    }

    // chunk_key_encoding
    {
        JsonObject cke;
        cke["name"] = JsonValue(meta.chunk_key_encoding);
        if (meta.chunk_key_encoding == "v2") {
            JsonObject cke_cfg;
            cke_cfg["separator"] = JsonValue(meta.dimension_separator);
            cke["configuration"] = JsonValue(std::move(cke_cfg));
        }
        root["chunk_key_encoding"] = JsonValue(std::move(cke));
    }

    // fill_value
    if (meta.fill_value.has_value())
        root["fill_value"] = JsonValue(*meta.fill_value);
    else
        root["fill_value"] = JsonValue(nullptr);

    // codecs
    {
        JsonArray codecs_arr;

        // If we have a shard_config but no codecs list was provided, build one.
        if (meta.codecs.empty() && meta.shard_config) {
            // Build sharding_indexed codec entry.
            JsonObject sc_cfg;
            {
                JsonArray sub_cs;
                for (auto c : meta.shard_config->sub_chunks)
                    sub_cs.push_back(JsonValue(c));
                sc_cfg["chunk_shape"] = JsonValue(std::move(sub_cs));
            }
            sc_cfg["index_location"] = JsonValue("start");

            {
                JsonArray idx_codecs;
                for (const auto& ic : meta.shard_config->index_codecs)
                    idx_codecs.push_back(codec_config_to_json(ic));
                if (idx_codecs.empty()) {
                    // Default: bytes codec for index.
                    JsonObject bytes_codec;
                    bytes_codec["name"] = JsonValue("bytes");
                    JsonObject bytes_cfg;
                    bytes_cfg["endian"] = JsonValue("little");
                    bytes_codec["configuration"] = JsonValue(std::move(bytes_cfg));
                    idx_codecs.push_back(JsonValue(std::move(bytes_codec)));
                }
                sc_cfg["index_codecs"] = JsonValue(std::move(idx_codecs));
            }

            {
                JsonArray sub_codecs;
                for (const auto& sc : meta.shard_config->sub_codecs)
                    sub_codecs.push_back(codec_config_to_json(sc));
                if (sub_codecs.empty()) {
                    JsonObject bytes_codec;
                    bytes_codec["name"] = JsonValue("bytes");
                    JsonObject bytes_cfg;
                    bytes_cfg["endian"] = JsonValue("little");
                    bytes_codec["configuration"] = JsonValue(std::move(bytes_cfg));
                    sub_codecs.push_back(JsonValue(std::move(bytes_codec)));
                }
                sc_cfg["codecs"] = JsonValue(std::move(sub_codecs));
            }

            JsonObject sharding_obj;
            sharding_obj["name"] = JsonValue("sharding_indexed");
            sharding_obj["configuration"] = JsonValue(std::move(sc_cfg));
            codecs_arr.push_back(JsonValue(std::move(sharding_obj)));
        } else if (meta.codecs.empty()) {
            // Default: bytes codec.
            JsonObject bytes_codec;
            bytes_codec["name"] = JsonValue("bytes");
            JsonObject bytes_cfg;
            bytes_cfg["endian"] = JsonValue("little");
            bytes_codec["configuration"] = JsonValue(std::move(bytes_cfg));
            codecs_arr.push_back(JsonValue(std::move(bytes_codec)));
        } else {
            for (const auto& cc : meta.codecs)
                codecs_arr.push_back(codec_config_to_json(cc));
        }

        root["codecs"] = JsonValue(std::move(codecs_arr));
    }

    return json_serialize(JsonValue(std::move(root)), 2) + "\n";
}

// ----- ConsolidatedMetadata::parse -----

ConsolidatedMetadata ConsolidatedMetadata::parse(std::string_view json_str) {
    auto root = json_parse(json_str);
    ConsolidatedMetadata cm;
    if (!root.is_object()) return cm;

    auto* meta = json_find(root, "metadata");
    if (!meta || !meta->is_object()) return cm;

    for (auto it = meta->begin(); it != meta->end(); ++it) {
        auto key = it.key();
        const auto& val = *it;
        if (key.size() >= 7 && key.substr(key.size() - 7) == ".zarray") {
            // This is an array metadata entry.
            auto array_path = key.substr(0, key.size() - 8); // strip "/.zarray"
            if (!array_path.empty() && array_path.front() == '/')
                array_path = array_path.substr(1);
            auto json_text = json_serialize(val);
            cm.arrays[array_path] = parse_zarray(json_text);
        } else if (key.size() >= 7 && key.substr(key.size() - 7) == ".zattrs") {
            auto attr_path = key.substr(0, key.size() - 8);
            if (!attr_path.empty() && attr_path.front() == '/')
                attr_path = attr_path.substr(1);
            cm.attrs[attr_path] = val;
        }
    }
    return cm;
}

} // namespace detail

// ---------------------------------------------------------------------------
// OME-NGFF metadata parsing
// ---------------------------------------------------------------------------

namespace ome_detail {

std::vector<CoordinateTransform>
parse_transforms(const JsonValue& arr) {
    std::vector<CoordinateTransform> out;
    if (!arr.is_array()) return out;
    for (const auto& t : arr) {
        if (!t.is_object()) continue;
        auto* tp = json_find(t, "type");
        if (!tp || !tp->is_string()) continue;
        const auto& type_str = tp->get_string();
        if (type_str == "scale") {
            ScaleTransform st;
            if (auto* s = json_find(t, "scale"); s && s->is_array()) {
                for (const auto& v : (*s))
                    st.scale.push_back(v.get_double());
            }
            out.emplace_back(std::move(st));
        } else if (type_str == "translation") {
            TranslationTransform tt;
            if (auto* s = json_find(t, "translation"); s && s->is_array()) {
                for (const auto& v : (*s))
                    tt.translation.push_back(v.get_double());
            }
            out.emplace_back(std::move(tt));
        }
    }
    return out;
}

JsonValue serialize_transforms(
    const std::vector<CoordinateTransform>& transforms) {
    JsonArray arr;
    for (const auto& t : transforms) {
        if (auto* st = std::get_if<ScaleTransform>(&t)) {
            JsonArray vals;
            for (double v : st->scale) vals.emplace_back(v);
            arr.push_back(json_object({
                {"type", "scale"},
                {"scale", JsonValue{std::move(vals)}}
            }));
        } else if (auto* tt = std::get_if<TranslationTransform>(&t)) {
            JsonArray vals;
            for (double v : tt->translation) vals.emplace_back(v);
            arr.push_back(json_object({
                {"type", "translation"},
                {"translation", JsonValue{std::move(vals)}}
            }));
        }
    }
    return JsonValue{std::move(arr)};
}

} // namespace ome_detail

// ---------------------------------------------------------------------------
// parse_ome_metadata
// ---------------------------------------------------------------------------

MultiscaleMetadata parse_ome_metadata(const JsonValue& attrs) {
    MultiscaleMetadata meta;

    const JsonValue* ms_arr = json_find(attrs, "multiscales");
    if (!ms_arr || !ms_arr->is_array() || (*ms_arr).empty())
        throw std::runtime_error("zarr: missing or empty 'multiscales' in .zattrs");

    const auto& ms = (*ms_arr)[0];
    if (!ms.is_object())
        throw std::runtime_error("zarr: multiscales entry must be an object");

    if (auto* p = json_find(ms, "version"); p && p->is_string())
        meta.version = p->get_string();
    if (auto* p = json_find(ms, "name"); p && p->is_string())
        meta.name = p->get_string();
    if (auto* p = json_find(ms, "type"); p && p->is_string())
        meta.type = p->get_string();

    if (auto* ax = json_find(ms, "axes"); ax && ax->is_array()) {
        for (const auto& a : (*ax)) {
            Axis axis;
            if (auto* n = json_find(a, "name"); n && n->is_string())
                axis.name = n->get_string();
            if (auto* t = json_find(a, "type"); t && t->is_string())
                axis.type = ome_detail::parse_axis_type(t->get_string());
            if (auto* u = json_find(a, "unit"); u && u->is_string())
                axis.unit = u->get_string();
            meta.axes.push_back(std::move(axis));
        }
    }

    if (auto* ds_arr = json_find(ms, "datasets"); ds_arr && ds_arr->is_array()) {
        for (const auto& ds : (*ds_arr)) {
            MultiscaleDataset dataset;
            if (auto* p = json_find(ds, "path"); p && p->is_string())
                dataset.path = p->get_string();
            if (auto* ct = json_find(ds, "coordinateTransformations"))
                dataset.transforms = ome_detail::parse_transforms(*ct);
            meta.datasets.push_back(std::move(dataset));
        }
    }

    return meta;
}

// ---------------------------------------------------------------------------
// serialize_ome_metadata
// ---------------------------------------------------------------------------

JsonValue serialize_ome_metadata(const MultiscaleMetadata& meta) {
    JsonArray axes_arr;
    for (const auto& ax : meta.axes) {
        JsonObject obj;
        obj["name"] = JsonValue{ax.name};
        obj["type"] = JsonValue{std::string(ome_detail::axis_type_string(ax.type))};
        if (!ax.unit.empty())
            obj["unit"] = JsonValue{ax.unit};
        axes_arr.push_back(JsonValue{std::move(obj)});
    }

    JsonArray ds_arr;
    for (const auto& ds : meta.datasets) {
        JsonObject obj;
        obj["path"] = JsonValue{ds.path};
        obj["coordinateTransformations"] =
            ome_detail::serialize_transforms(ds.transforms);
        ds_arr.push_back(JsonValue{std::move(obj)});
    }

    JsonObject ms;
    ms["version"] = JsonValue{meta.version};
    if (!meta.name.empty())
        ms["name"] = JsonValue{meta.name};
    if (!meta.type.empty())
        ms["type"] = JsonValue{meta.type};
    ms["axes"] = JsonValue{std::move(axes_arr)};
    ms["datasets"] = JsonValue{std::move(ds_arr)};

    JsonArray ms_arr;
    ms_arr.push_back(JsonValue{std::move(ms)});

    return json_object({{"multiscales", JsonValue{std::move(ms_arr)}}});
}

// ---------------------------------------------------------------------------
// parse_label_metadata
// ---------------------------------------------------------------------------

LabelMetadata parse_label_metadata(const JsonValue& attrs) {
    LabelMetadata meta;

    if (auto* p = json_find(attrs, "image-label")) {
        if (!p->is_object())
            throw std::runtime_error("zarr: 'image-label' must be an object");

        if (auto* v = json_find(*p, "version"); v && v->is_string())
            meta.version = v->get_string();

        if (auto* colors = json_find(*p, "colors"); colors && colors->is_array()) {
            for (const auto& c : (*colors)) {
                LabelColor lc{};
                if (auto* lv = json_find(c, "label-value"); lv && lv->is_number())
                    lc.label_value = static_cast<std::uint32_t>(lv->get_int());
                if (auto* rgba = json_find(c, "rgba"); rgba && rgba->is_array()) {
                    const auto& arr = (*rgba);
                    for (std::size_t i = 0; i < 4 && i < arr.size(); ++i)
                        lc.rgba[i] = static_cast<std::uint8_t>(arr[i].get_int());
                }
                meta.colors.push_back(lc);
            }
        }

        if (auto* props = json_find(*p, "properties"); props && props->is_array()) {
            for (const auto& prop : (*props)) {
                if (prop.is_string())
                    meta.properties.push_back(prop.get_string());
            }
        }
    }

    return meta;
}

// ---------------------------------------------------------------------------
// parse_plate_metadata
// ---------------------------------------------------------------------------

PlateMetadata parse_plate_metadata(const JsonValue& attrs) {
    PlateMetadata meta;

    const JsonValue* plate = json_find(attrs, "plate");
    if (!plate || !plate->is_object())
        throw std::runtime_error("zarr: missing or invalid 'plate' in .zattrs");

    if (auto* v = json_find(*plate, "version"); v && v->is_string())
        meta.version = v->get_string();
    if (auto* n = json_find(*plate, "name"); n && n->is_string())
        meta.name = n->get_string();
    if (auto* fc = json_find(*plate, "field_count"); fc && fc->is_number())
        meta.field_count = fc->get_size_t();

    if (auto* cols = json_find(*plate, "columns"); cols && cols->is_array()) {
        for (const auto& c : (*cols)) {
            if (auto* n = json_find(c, "name"); n && n->is_string())
                meta.columns.push_back(n->get_string());
        }
    }

    if (auto* rows = json_find(*plate, "rows"); rows && rows->is_array()) {
        for (const auto& r : (*rows)) {
            if (auto* n = json_find(r, "name"); n && n->is_string())
                meta.rows.push_back(n->get_string());
        }
    }

    if (auto* wells = json_find(*plate, "wells"); wells && wells->is_array()) {
        for (const auto& w : (*wells)) {
            WellRef ref;
            if (auto* p = json_find(w, "path"); p && p->is_string())
                ref.path = p->get_string();
            if (auto* r = json_find(w, "rowIndex"); r && r->is_number())
                ref.row = r->get_size_t();
            if (auto* c = json_find(w, "columnIndex"); c && c->is_number())
                ref.col = c->get_size_t();
            meta.wells.push_back(std::move(ref));
        }
    }

    return meta;
}

// ---------------------------------------------------------------------------
// load_consolidated_metadata
// ---------------------------------------------------------------------------

detail::ConsolidatedMetadata
load_consolidated_metadata(const std::filesystem::path& root) {
    auto zmetadata_path = root / ".zmetadata";
    auto json = detail::read_file(zmetadata_path);
    return detail::ConsolidatedMetadata::parse(json);
}

// ---------------------------------------------------------------------------
// ZarrArray methods moved from header (need full json.hpp)
// ---------------------------------------------------------------------------

std::string ZarrArray::read_attrs() const {
    if (store_) {
        auto key = [&](const std::string& name) -> std::string {
            return array_key_.empty() ? name : array_key_ + "/" + name;
        };
        if (meta_.version == ZarrVersion::v3) {
            auto data = store_->get_if_exists(key("zarr.json"));
            if (!data) return "{}";
            std::string str(reinterpret_cast<const char*>(data->data()), data->size());
            auto root = json_parse(str);
            if (auto* p = json_find(root, "attributes"); p)
                return json_serialize(*p, 2);
            return "{}";
        }
        auto data = store_->get_if_exists(key(".zattrs"));
        if (!data) return "{}";
        return std::string(reinterpret_cast<const char*>(data->data()), data->size());
    }
    if (meta_.version == ZarrVersion::v3) {
        auto zj_path = root_ / "zarr.json";
        if (!std::filesystem::exists(zj_path)) return "{}";
        auto json = detail::read_file(zj_path);
        auto root = json_parse(json);
        if (auto* p = json_find(root, "attributes"); p)
            return json_serialize(*p, 2);
        return "{}";
    }
    auto p = root_ / ".zattrs";
    if (!std::filesystem::exists(p)) return "{}";
    return detail::read_file(p);
}

void ZarrArray::write_attrs(std::string_view json) {
    if (store_) {
        auto key = [&](const std::string& name) -> std::string {
            return array_key_.empty() ? name : array_key_ + "/" + name;
        };
        if (meta_.version == ZarrVersion::v3) {
            auto data = store_->get_if_exists(key("zarr.json"));
            std::string zj_str = data
                ? std::string(reinterpret_cast<const char*>(data->data()), data->size())
                : "{}";
            auto root = json_parse(zj_str);
            auto attrs = json_parse(json);
            root["attributes"] = std::move(attrs);
            auto out = json_serialize(root, 2) + "\n";
            auto bytes = std::as_bytes(std::span{out});
            store_->set(key("zarr.json"), {bytes.data(), bytes.size()});
        } else {
            auto bytes = std::as_bytes(std::span{json});
            store_->set(key(".zattrs"), {bytes.data(), bytes.size()});
        }
        return;
    }
    if (meta_.version == ZarrVersion::v3) {
        auto zj_path = root_ / "zarr.json";
        auto zj_str = detail::read_file(zj_path);
        auto root = json_parse(zj_str);
        auto attrs = json_parse(json);
        root["attributes"] = std::move(attrs);
        detail::write_file(zj_path, json_serialize(root, 2) + "\n");
    } else {
        detail::write_file(root_ / ".zattrs", json);
    }
}

bool ZarrArray::needs_byteswap() const noexcept {
    auto elem_sz = dtype_size(meta_.dtype);
    if (elem_sz <= 1) return false;
    bool native_le = detail::is_little_endian();
    if (meta_.version == ZarrVersion::v2) {
        if (meta_.byte_order == '|') return false;
        return (meta_.byte_order == '<') != native_le;
    }
    for (const auto& cc : meta_.codecs) {
        if (cc.name == "bytes" && cc.configuration && cc.configuration->is_object()) {
            auto* e = json_find(*cc.configuration, "endian");
            if (e && e->is_string()) {
                bool stored_le = (e->get_string() == "little");
                return stored_le != native_le;
            }
        }
    }
    return false;
}

std::optional<std::vector<std::byte>>
ZarrArray::extract_inner_chunk(std::span<const std::byte> shard_data,
                               std::span<const std::size_t> inner_indices) const {
    const auto& sc = *meta_.shard_config;
    const auto n_inner = meta_.total_sub_chunks_per_shard();
    const std::size_t index_size = n_inner * 16;

    if (shard_data.size() < index_size)
        throw std::runtime_error("zarr: shard too small to contain index");

    // Index is always at the start of the shard.
    std::span<const std::byte> index_data = shard_data.subspan(0, index_size);

    std::vector<std::byte> decoded_index;
    if (!sc.index_codecs.empty()) {
        decoded_index.assign(index_data.begin(), index_data.end());
        for (const auto& ic : sc.index_codecs) {
            if (ic.name == "bytes") {
                if (ic.configuration && ic.configuration->is_object()) {
                    auto* e = json_find(*ic.configuration, "endian");
                    if (e && e->is_string() &&
                        e->get_string() == "big" &&
                        detail::is_little_endian()) {
                        detail::byteswap_inplace(decoded_index, 8);
                    } else if (e && e->is_string() &&
                               e->get_string() == "little" &&
                               !detail::is_little_endian()) {
                        detail::byteswap_inplace(decoded_index, 8);
                    }
                }
            } else if (codec_.decompress) {
                auto it = registry_.find(ic.name);
                if (it != registry_.end() && it->second.decompress) {
                    decoded_index = it->second.decompress(
                        decoded_index, n_inner * 16);
                }
            }
        }
        index_data = decoded_index;
    }

    auto index = detail::ShardIndex::deserialize(index_data, n_inner);

    std::size_t linear = 0;
    std::size_t stride = 1;
    for (std::size_t d = inner_indices.size(); d-- > 0;) {
        linear += inner_indices[d] * stride;
        stride *= meta_.sub_chunks_per_shard(d);
    }

    if (linear >= n_inner)
        throw std::runtime_error("zarr: inner chunk index out of range");

    const auto& entry = index.entries[linear];
    if (entry.is_missing()) return std::nullopt;

    if (entry.offset + entry.nbytes > shard_data.size())
        throw std::runtime_error("zarr: inner chunk offset/size exceeds shard data");

    std::vector<std::byte> chunk(
        shard_data.begin() + static_cast<std::ptrdiff_t>(entry.offset),
        shard_data.begin() + static_cast<std::ptrdiff_t>(entry.offset + entry.nbytes));

    if (codec_.decompress && needs_decompression()) {
        chunk = codec_.decompress(chunk, meta_.sub_chunk_byte_size());
    }

    if (needs_byteswap()) {
        detail::byteswap_inplace(chunk, dtype_size(meta_.dtype));
    }

    return chunk;
}

// ---------------------------------------------------------------------------
// OmeZarrReader methods moved from header
// ---------------------------------------------------------------------------

OmeZarrReader::OmeZarrReader(const std::filesystem::path& root)
    : root_(root) {
    auto zattrs_path = root_ / ".zattrs";
    if (!std::filesystem::exists(zattrs_path))
        throw std::runtime_error("zarr: missing .zattrs at " + root_.string());

    auto json_str = detail::read_file(zattrs_path);
    attrs_ = std::make_shared<JsonValue>(json_parse(json_str));
    meta_ = parse_ome_metadata(*attrs_);

    levels_.reserve(meta_.datasets.size());
    for (const auto& ds : meta_.datasets) {
        levels_.push_back(ZarrArray::open(root_ / ds.path));
    }
}

std::vector<std::string> OmeZarrReader::label_names() const {
    std::vector<std::string> names;
    auto labels_dir = root_ / "labels";
    if (!std::filesystem::is_directory(labels_dir)) return names;

    auto zattrs_path = labels_dir / ".zattrs";
    if (std::filesystem::exists(zattrs_path)) {
        auto json_str = detail::read_file(zattrs_path);
        auto attrs = json_parse(json_str);
        if (auto* p = json_find(attrs, "labels"); p && p->is_array()) {
            for (const auto& v : (*p)) {
                if (v.is_string())
                    names.push_back(v.get_string());
            }
            return names;
        }
    }

    for (const auto& entry : std::filesystem::directory_iterator(labels_dir)) {
        if (entry.is_directory())
            names.push_back(entry.path().filename().string());
    }
    return names;
}

bool OmeZarrReader::is_plate() const noexcept {
    return attrs_ && json_find(*attrs_, "plate") != nullptr;
}

std::optional<PlateMetadata> OmeZarrReader::plate_metadata() const {
    if (!is_plate()) return std::nullopt;
    return parse_plate_metadata(*attrs_);
}

// ---------------------------------------------------------------------------
// OmeZarrWriter methods moved from header
// ---------------------------------------------------------------------------

void OmeZarrWriter::finalize() {
    detail::write_file(config_.root / ".zgroup",
                       "{\"zarr_format\": 2}\n");

    auto attrs = serialize_ome_metadata(config_.multiscale);
    detail::write_file(config_.root / ".zattrs",
                       json_serialize(attrs, 2) + "\n");
}

ZarrArray& OmeZarrWriter::add_level(std::vector<std::size_t> shape) {
    std::size_t idx = levels_.size();
    if (idx >= config_.multiscale.datasets.size())
        throw std::runtime_error("zarr: too many levels for metadata");

    const auto& ds = config_.multiscale.datasets[idx];
    auto level_path = config_.root / ds.path;

    ZarrMetadata meta;
    meta.shape = std::move(shape);
    meta.chunks = config_.chunk_shape;
    meta.dtype = config_.dtype;
    meta.dimension_separator = "/";

    levels_.push_back(ZarrArray::create(level_path, std::move(meta), config_.codec));
    return levels_.back();
}

// ---------------------------------------------------------------------------
// ZarrFilter
// ---------------------------------------------------------------------------

std::vector<std::byte> ZarrFilter::encode(std::span<const std::byte> input) const {
    switch (id) {
        case ZarrFilterId::delta:           return encode_delta(input);
        case ZarrFilterId::fixedscaleoffset: return encode_fixedscaleoffset(input);
        case ZarrFilterId::quantize:        return encode_quantize(input);
    }
    return {input.begin(), input.end()};
}

std::vector<std::byte> ZarrFilter::decode(std::span<const std::byte> input) const {
    switch (id) {
        case ZarrFilterId::delta:           return decode_delta(input);
        case ZarrFilterId::fixedscaleoffset: return decode_fixedscaleoffset(input);
        case ZarrFilterId::quantize:        return {input.begin(), input.end()}; // quantize is lossy, decode is identity
    }
    return {input.begin(), input.end()};
}

std::vector<std::byte> ZarrFilter::encode_delta(std::span<const std::byte> input) const {
    const auto elem_sz = dtype_size(dtype);
    // Only uint8 (1 byte) and uint16 (2 bytes) are supported.
    if ((elem_sz != 1 && elem_sz != 2) || input.size() % elem_sz != 0)
        return {input.begin(), input.end()};

    const auto n = input.size() / elem_sz;
    std::vector<std::byte> out(input.size());
    // First element is stored as-is.
    std::memcpy(out.data(), input.data(), elem_sz);
    // Subsequent elements store the delta on whole elements.
    if (elem_sz == 1) {
        auto* src = reinterpret_cast<const std::uint8_t*>(input.data());
        auto* dst = reinterpret_cast<std::uint8_t*>(out.data());
        for (std::size_t i = 1; i < n; ++i)
            dst[i] = static_cast<std::uint8_t>(src[i] - src[i - 1]);
    } else {
        auto* src = reinterpret_cast<const std::uint16_t*>(input.data());
        auto* dst = reinterpret_cast<std::uint16_t*>(out.data());
        for (std::size_t i = 1; i < n; ++i)
            dst[i] = static_cast<std::uint16_t>(src[i] - src[i - 1]);
    }
    return out;
}

std::vector<std::byte> ZarrFilter::decode_delta(std::span<const std::byte> input) const {
    const auto elem_sz = dtype_size(dtype);
    // Only uint8 (1 byte) and uint16 (2 bytes) are supported.
    if ((elem_sz != 1 && elem_sz != 2) || input.size() % elem_sz != 0)
        return {input.begin(), input.end()};

    const auto n = input.size() / elem_sz;
    std::vector<std::byte> out(input.size());
    std::memcpy(out.data(), input.data(), elem_sz);
    if (elem_sz == 1) {
        auto* src = reinterpret_cast<const std::uint8_t*>(input.data());
        auto* dst = reinterpret_cast<std::uint8_t*>(out.data());
        for (std::size_t i = 1; i < n; ++i)
            dst[i] = static_cast<std::uint8_t>(src[i] + dst[i - 1]);
    } else {
        auto* src = reinterpret_cast<const std::uint16_t*>(input.data());
        auto* dst = reinterpret_cast<std::uint16_t*>(out.data());
        for (std::size_t i = 1; i < n; ++i)
            dst[i] = static_cast<std::uint16_t>(src[i] + dst[i - 1]);
    }
    return out;
}

std::vector<std::byte> ZarrFilter::encode_fixedscaleoffset(std::span<const std::byte> input) const {
    // Simplified: applies (x - offset) * scale element-wise on float64 data.
    const auto elem_sz = dtype_size(dtype);
    if (elem_sz == 0 || input.size() % elem_sz != 0)
        return {input.begin(), input.end()};
    if (dtype != ZarrDtype::float64 && dtype != ZarrDtype::float32)
        return {input.begin(), input.end()};

    std::vector<std::byte> out(input.size());
    const auto n = input.size() / elem_sz;
    if (dtype == ZarrDtype::float64) {
        for (std::size_t i = 0; i < n; ++i) {
            double val;
            std::memcpy(&val, input.data() + i * elem_sz, sizeof(double));
            val = (val - offset) * scale;
            std::memcpy(out.data() + i * elem_sz, &val, sizeof(double));
        }
    } else {
        for (std::size_t i = 0; i < n; ++i) {
            float val;
            std::memcpy(&val, input.data() + i * elem_sz, sizeof(float));
            val = static_cast<float>((val - offset) * scale);
            std::memcpy(out.data() + i * elem_sz, &val, sizeof(float));
        }
    }
    return out;
}

std::vector<std::byte> ZarrFilter::decode_fixedscaleoffset(std::span<const std::byte> input) const {
    const auto elem_sz = dtype_size(dtype);
    if (elem_sz == 0 || input.size() % elem_sz != 0)
        return {input.begin(), input.end()};
    if (dtype != ZarrDtype::float64 && dtype != ZarrDtype::float32)
        return {input.begin(), input.end()};

    std::vector<std::byte> out(input.size());
    const auto n = input.size() / elem_sz;
    if (dtype == ZarrDtype::float64) {
        for (std::size_t i = 0; i < n; ++i) {
            double val;
            std::memcpy(&val, input.data() + i * elem_sz, sizeof(double));
            val = val / scale + offset;
            std::memcpy(out.data() + i * elem_sz, &val, sizeof(double));
        }
    } else {
        for (std::size_t i = 0; i < n; ++i) {
            float val;
            std::memcpy(&val, input.data() + i * elem_sz, sizeof(float));
            val = static_cast<float>(val / scale + offset);
            std::memcpy(out.data() + i * elem_sz, &val, sizeof(float));
        }
    }
    return out;
}

std::vector<std::byte> ZarrFilter::encode_quantize(std::span<const std::byte> input) const {
    const auto elem_sz = dtype_size(dtype);
    if (elem_sz == 0 || input.size() % elem_sz != 0)
        return {input.begin(), input.end()};

    std::vector<std::byte> out(input.size());
    const auto n = input.size() / elem_sz;
    const double factor = std::pow(10.0, digits);
    if (dtype == ZarrDtype::float64) {
        for (std::size_t i = 0; i < n; ++i) {
            double val;
            std::memcpy(&val, input.data() + i * elem_sz, sizeof(double));
            val = std::round(val * factor) / factor;
            std::memcpy(out.data() + i * elem_sz, &val, sizeof(double));
        }
    } else if (dtype == ZarrDtype::float32) {
        for (std::size_t i = 0; i < n; ++i) {
            float val;
            std::memcpy(&val, input.data() + i * elem_sz, sizeof(float));
            val = static_cast<float>(std::round(val * factor) / factor);
            std::memcpy(out.data() + i * elem_sz, &val, sizeof(float));
        }
    } else {
        return {input.begin(), input.end()};
    }
    return out;
}

// ---------------------------------------------------------------------------
// is_canonical_c3d
// ---------------------------------------------------------------------------

bool is_canonical_c3d(const ZarrMetadata& m) noexcept {
    if (m.version != ZarrVersion::v3) return false;
    if (!m.shard_config) return false;
    const auto& sc = *m.shard_config;
    if (sc.sub_chunks.size() < 3) return false;
    if (sc.sub_chunks[0] != 256 || sc.sub_chunks[1] != 256 || sc.sub_chunks[2] != 256)
        return false;
    if (m.chunks.size() < 3) return false;
    for (int d = 0; d < 3; ++d) if (m.chunks[d] % 256 != 0) return false;
    if (m.dtype != ZarrDtype::uint8) return false;
    return true;
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

std::optional<std::vector<std::byte>>
Store::get_partial(const std::string& key, std::size_t offset, std::size_t length) const {
    auto data = get_if_exists(key);
    if (!data) return std::nullopt;
    if (offset >= data->size()) return std::vector<std::byte>{};
    auto end = std::min(offset + length, data->size());
    return std::vector<std::byte>(data->begin() + static_cast<std::ptrdiff_t>(offset),
                                   data->begin() + static_cast<std::ptrdiff_t>(end));
}

// ---------------------------------------------------------------------------
// FileSystemStore
// ---------------------------------------------------------------------------

bool FileSystemStore::exists(const std::string& key) const {
    return std::filesystem::exists(safe_path(key));
}

std::vector<std::byte> FileSystemStore::get(const std::string& key) const {
    auto p = safe_path(key);
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("zarr store: cannot open: " + p.string());
    auto sz = f.tellg();
    f.seekg(0);
    std::vector<std::byte> buf(static_cast<std::size_t>(sz));
    f.read(reinterpret_cast<char*>(buf.data()), sz);
    return buf;
}

std::optional<std::vector<std::byte>>
FileSystemStore::get_if_exists(const std::string& key) const {
    auto p = safe_path(key);
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) return std::nullopt;
    auto sz = f.tellg();
    f.seekg(0);
    std::vector<std::byte> buf(static_cast<std::size_t>(sz));
    f.read(reinterpret_cast<char*>(buf.data()), sz);
    return buf;
}

std::optional<std::vector<std::byte>>
FileSystemStore::get_partial(const std::string& key, std::size_t offset, std::size_t length) const {
    auto p = safe_path(key);
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) return std::nullopt;
    auto file_sz = static_cast<std::size_t>(f.tellg());
    if (offset >= file_sz) return std::vector<std::byte>{};
    auto end = std::min(offset + length, file_sz);
    auto actual_len = end - offset;
    f.seekg(static_cast<std::streamoff>(offset));
    std::vector<std::byte> buf(actual_len);
    f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(actual_len));
    return buf;
}

void FileSystemStore::set(const std::string& key, std::span<const std::byte> value) {
    auto p = safe_path(key);
    std::filesystem::create_directories(p.parent_path());
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("zarr store: cannot write: " + p.string());
    f.write(reinterpret_cast<const char*>(value.data()),
            static_cast<std::streamsize>(value.size()));
}

void FileSystemStore::erase(const std::string& key) {
    std::filesystem::remove(safe_path(key));
}

std::filesystem::path FileSystemStore::safe_path(const std::string& key) const {
    auto p = (root_ / key).lexically_normal();
    // Ensure the resolved path is within root_ (prevent ../ traversal).
    auto root_norm = root_.lexically_normal();
    auto [root_end, p_begin] = std::mismatch(
        root_norm.begin(), root_norm.end(), p.begin(), p.end());
    if (root_end != root_norm.end())
        throw std::runtime_error("zarr store: path traversal rejected: " + key);
    return p;
}

// ---------------------------------------------------------------------------
// HttpStore
// ---------------------------------------------------------------------------

HttpStore::HttpStore(std::string base_url)
    : base_url_(strip_trailing_slash(std::move(base_url)))
    , client_(std::make_shared<HttpClient>()) {}

HttpStore::HttpStore(std::string base_url, HttpClient::Config config)
    : base_url_(strip_trailing_slash(std::move(base_url)))
    , client_(std::make_shared<HttpClient>(std::move(config))) {}

HttpStore::HttpStore(std::string base_url, AwsAuth auth)
    : base_url_(strip_trailing_slash(std::move(base_url)))
{
    HttpClient::Config cfg;
    cfg.aws_auth = std::move(auth);
    cfg.transfer_timeout = std::chrono::seconds{60};
    client_ = std::make_shared<HttpClient>(std::move(cfg));
}

bool HttpStore::exists(const std::string& key) const {
    return client_->head(make_url(key)).ok();
}

std::vector<std::byte> HttpStore::get(const std::string& key) const {
    auto data = get_if_exists(key);
    if (!data)
        throw std::runtime_error("HttpStore: key not found: " + key);
    return std::move(*data);
}

std::optional<std::vector<std::byte>>
HttpStore::get_if_exists(const std::string& key) const {
    auto resp = client_->get(make_url(key));
    if (!resp.ok()) return std::nullopt;
    return std::move(resp.body);
}

void HttpStore::set(const std::string& /*key*/, std::span<const std::byte> /*value*/) {
    throw std::runtime_error("HttpStore is read-only");
}

void HttpStore::erase(const std::string& /*key*/) {
    throw std::runtime_error("HttpStore is read-only");
}

std::optional<std::vector<std::byte>>
HttpStore::get_partial(const std::string& key, std::size_t offset, std::size_t length) const {
    auto resp = client_->get_range(make_url(key), offset, length);
    if (!resp.ok()) return std::nullopt;
    return std::move(resp.body);
}

std::string HttpStore::make_url(const std::string& key) const {
    return base_url_ + "/" + key;
}

std::string HttpStore::strip_trailing_slash(std::string s) {
    while (!s.empty() && s.back() == '/')
        s.pop_back();
    return s;
}

// ---------------------------------------------------------------------------
// detail -- I/O and metadata helpers
// ---------------------------------------------------------------------------

namespace detail {

std::string read_file(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("zarr: cannot open file: " + p.string());
    auto sz = f.tellg();
    f.seekg(0);
    std::string buf(static_cast<std::size_t>(sz), '\0');
    f.read(buf.data(), sz);
    return buf;
}

std::vector<std::byte> read_file_bytes(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("zarr: cannot open file: " + p.string());
    auto sz = f.tellg();
    f.seekg(0);
    std::vector<std::byte> buf(static_cast<std::size_t>(sz));
    f.read(reinterpret_cast<char*>(buf.data()), sz);
    return buf;
}

void write_file(const std::filesystem::path& p, std::string_view data) {
    auto tmp = p;
    tmp += ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) throw std::runtime_error("zarr: cannot write file: " + p.string());
        f.write(data.data(), static_cast<std::streamsize>(data.size()));
    }
    std::filesystem::rename(tmp, p);
}

void write_file_bytes(const std::filesystem::path& p, std::span<const std::byte> data) {
    // Atomic write: write to .tmp, then rename. Prevents corrupt files
    // if the process is interrupted (e.g. curl abort during shutdown).
    auto tmp = p;
    tmp += ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) throw std::runtime_error("zarr: cannot write file: " + tmp.string());
        f.write(reinterpret_cast<const char*>(data.data()),
                static_cast<std::streamsize>(data.size()));
    }
    std::filesystem::rename(tmp, p);
}

bool is_little_endian() noexcept {
    const std::uint32_t one = 1;
    return *reinterpret_cast<const std::uint8_t*>(&one) == 1;
}

void byteswap_inplace(std::span<std::byte> data, std::size_t elem_size) {
    if (elem_size <= 1) return;
    for (std::size_t i = 0; i + elem_size <= data.size(); i += elem_size) {
        std::reverse(data.begin() + static_cast<std::ptrdiff_t>(i),
                     data.begin() + static_cast<std::ptrdiff_t>(i + elem_size));
    }
}

void write_le64(std::byte* dst, std::uint64_t val) {
    for (int i = 0; i < 8; ++i)
        dst[i] = static_cast<std::byte>((val >> (8 * i)) & 0xFF);
}

std::uint64_t read_le64(const std::byte* src) {
    std::uint64_t val = 0;
    for (int i = 0; i < 8; ++i)
        val |= static_cast<std::uint64_t>(static_cast<std::uint8_t>(src[i])) << (8 * i);
    return val;
}

std::vector<std::byte> ShardIndex::serialize() const {
    std::vector<std::byte> buf(entries.size() * 16);
    for (std::size_t i = 0; i < entries.size(); ++i) {
        write_le64(buf.data() + i * 16,     entries[i].offset);
        write_le64(buf.data() + i * 16 + 8, entries[i].nbytes);
    }
    return buf;
}

ShardIndex ShardIndex::deserialize(std::span<const std::byte> data, std::size_t num_chunks) {
    ShardIndex idx;
    idx.entries.resize(num_chunks);
    if (data.size() < num_chunks * 16)
        throw std::runtime_error("zarr: shard index too small");
    for (std::size_t i = 0; i < num_chunks; ++i) {
        idx.entries[i].offset = read_le64(data.data() + i * 16);
        idx.entries[i].nbytes  = read_le64(data.data() + i * 16 + 8);
    }
    return idx;
}

std::string serialize_zarray(const ZarrMetadata& meta) {
    std::string s = "{\n";
    s += "  \"zarr_format\": 2,\n";

    // shape
    s += "  \"shape\": [";
    for (std::size_t i = 0; i < meta.shape.size(); ++i) {
        if (i) s += ", ";
        s += std::to_string(meta.shape[i]);
    }
    s += "],\n";

    // chunks
    s += "  \"chunks\": [";
    for (std::size_t i = 0; i < meta.chunks.size(); ++i) {
        if (i) s += ", ";
        s += std::to_string(meta.chunks[i]);
    }
    s += "],\n";

    // dtype
    s += "  \"dtype\": \"";
    // Byte order is not applicable to single-byte dtypes; zarr v2 uses '|'.
    s += (dtype_size(meta.dtype) == 1) ? '|' : meta.byte_order;
    s += dtype_string_v2(meta.dtype);
    s += "\",\n";

    // compressor — emit a numcodecs-compatible config object (zarr v2 spec).
    // Field names differ per codec: blosc requires cname/clevel/shuffle/blocksize,
    // zstd/gzip/zlib use "level", lz4 uses "acceleration".
    if (meta.compressor_id.empty()) {
        s += "  \"compressor\": null,\n";
    } else if (meta.compressor_id == "blosc") {
        // Matches the encoder (VcDataset.cpp): cname=lz4, byte-shuffle, auto blocksize.
        s += "  \"compressor\": {\"id\": \"blosc\", \"cname\": \"lz4\", \"clevel\": "
             + std::to_string(meta.compression_level)
             + ", \"shuffle\": 1, \"blocksize\": 0},\n";
    } else if (meta.compressor_id == "lz4") {
        s += "  \"compressor\": {\"id\": \"lz4\", \"acceleration\": "
             + std::to_string(meta.compression_level) + "},\n";
    } else {
        // zstd, gzip, zlib, bz2, ...
        s += "  \"compressor\": {\"id\": \"" + meta.compressor_id + "\", \"level\": "
             + std::to_string(meta.compression_level) + "},\n";
    }

    // fill_value
    if (meta.fill_value.has_value()) {
        double fv = *meta.fill_value;
        if (fv == static_cast<double>(static_cast<std::int64_t>(fv)))
            s += "  \"fill_value\": " + std::to_string(static_cast<std::int64_t>(fv)) + ",\n";
        else
            s += "  \"fill_value\": " + std::to_string(fv) + ",\n";
    } else {
        s += "  \"fill_value\": null,\n";
    }

    s += "  \"order\": \"C\",\n";

    // filters
    if (meta.filters.empty()) {
        s += "  \"filters\": null,\n";
    } else {
        s += "  \"filters\": [";
        for (std::size_t i = 0; i < meta.filters.size(); ++i) {
            if (i) s += ", ";
            const auto& f = meta.filters[i];
            switch (f.id) {
                case ZarrFilterId::delta:
                    s += "{\"id\": \"delta\"";
                    s += ", \"dtype\": \"" + std::string(dtype_string_v2(f.dtype)) + "\"";
                    s += ", \"astype\": \"" + std::string(dtype_string_v2(f.astype)) + "\"";
                    s += "}";
                    break;
                case ZarrFilterId::fixedscaleoffset:
                    s += "{\"id\": \"fixedscaleoffset\"";
                    s += ", \"offset\": " + std::to_string(f.offset);
                    s += ", \"scale\": " + std::to_string(f.scale);
                    s += "}";
                    break;
                case ZarrFilterId::quantize:
                    s += "{\"id\": \"quantize\"";
                    s += ", \"digits\": " + std::to_string(f.digits);
                    s += "}";
                    break;
            }
        }
        s += "],\n";
    }

    s += "  \"dimension_separator\": \"" + meta.dimension_separator + "\"\n";
    s += "}\n";
    return s;
}

ZarrVersion detect_version(const std::filesystem::path& path) {
    if (std::filesystem::exists(path / "zarr.json"))
        return ZarrVersion::v3;
    if (std::filesystem::exists(path / ".zarray"))
        return ZarrVersion::v2;
    throw std::runtime_error("zarr: cannot detect version at: " + path.string() +
                             " (no .zarray or zarr.json found)");
}

} // namespace detail

// ---------------------------------------------------------------------------
// ShardBytes
// ---------------------------------------------------------------------------

ShardBytes ShardBytes::from_mmap(void* ptr, std::size_t n) noexcept {
    ShardBytes b;
    b.mapped_ = ptr;
    b.size_ = n;
    return b;
}

ShardBytes ShardBytes::from_vector(std::vector<std::byte> v) noexcept {
    ShardBytes b;
    b.size_ = v.size();
    b.vec_ = std::move(v);
    return b;
}

void ShardBytes::release() noexcept {
#if !defined(_WIN32)
    if (mapped_) {
        ::munmap(mapped_, size_);
    }
#endif
    mapped_ = nullptr;
    vec_.clear();
    vec_.shrink_to_fit();
    size_ = 0;
}

void ShardBytes::take(ShardBytes&& o) noexcept {
    mapped_ = o.mapped_; o.mapped_ = nullptr;
    vec_ = std::move(o.vec_);
    size_ = o.size_; o.size_ = 0;
}

// ---------------------------------------------------------------------------
// ZarrArray -- open / create
// ---------------------------------------------------------------------------

ZarrArray ZarrArray::open(const std::filesystem::path& path, Codec codec) {
    auto version = detail::detect_version(path);
    ZarrMetadata meta;
    if (version == ZarrVersion::v2) {
        auto json = detail::read_file(path / ".zarray");
        meta = detail::parse_zarray(json);
    } else {
        auto json = detail::read_file(path / "zarr.json");
        meta = detail::parse_zarr_json(json);
    }
    return ZarrArray(path, std::move(meta), std::move(codec));
}

ZarrArray ZarrArray::open(const std::filesystem::path& path, CodecRegistry registry) {
    auto version = detail::detect_version(path);
    ZarrMetadata meta;
    if (version == ZarrVersion::v2) {
        auto json = detail::read_file(path / ".zarray");
        meta = detail::parse_zarray(json);
    } else {
        auto json = detail::read_file(path / "zarr.json");
        meta = detail::parse_zarr_json(json);
    }

    // Try to find the appropriate codec from the registry.
    Codec codec;
    if (version == ZarrVersion::v2 && !meta.compressor_id.empty()) {
        auto it = registry.find(meta.compressor_id);
        if (it != registry.end()) codec = it->second;
    } else if (version == ZarrVersion::v3) {
        // Look for bytes-to-bytes codecs in the pipeline.  For sharded
        // arrays the outer pipeline only has sharding_indexed; the real
        // per-inner-chunk compressor lives in shard_config->sub_codecs.
        auto scan = [&](const std::vector<ZarrCodecConfig>& cs) {
            for (const auto& cc : cs) {
                if (cc.name != "bytes" && cc.name != "transpose"
                    && cc.name != "sharding_indexed") {
                    auto it = registry.find(cc.name);
                    if (it != registry.end()) { codec = it->second; return true; }
                }
            }
            return false;
        };
        if (meta.shard_config) scan(meta.shard_config->sub_codecs);
        if (!codec.decompress) scan(meta.codecs);
    }

    return ZarrArray(path, std::move(meta), std::move(codec), std::move(registry));
}

ZarrArray ZarrArray::create(const std::filesystem::path& path,
                            ZarrMetadata meta, Codec codec) {
    std::filesystem::create_directories(path);
    if (meta.version == ZarrVersion::v3) {
        auto json = detail::serialize_zarr_json(meta);
        detail::write_file(path / "zarr.json", json);
    } else {
        auto json = detail::serialize_zarray(meta);
        detail::write_file(path / ".zarray", json);
    }
    return ZarrArray(path, std::move(meta), std::move(codec));
}

ZarrArray ZarrArray::create(const std::filesystem::path& path,
                            ZarrMetadata meta, CodecRegistry registry) {
    Codec codec;
    if (meta.version == ZarrVersion::v2 && !meta.compressor_id.empty()) {
        auto it = registry.find(meta.compressor_id);
        if (it != registry.end()) codec = it->second;
    } else if (meta.version == ZarrVersion::v3) {
        for (const auto& cc : meta.codecs) {
            if (cc.name != "bytes" && cc.name != "transpose" && cc.name != "sharding_indexed") {
                auto it = registry.find(cc.name);
                if (it != registry.end()) { codec = it->second; break; }
            }
        }
    }

    std::filesystem::create_directories(path);
    if (meta.version == ZarrVersion::v3) {
        auto json = detail::serialize_zarr_json(meta);
        detail::write_file(path / "zarr.json", json);
    } else {
        auto json = detail::serialize_zarray(meta);
        detail::write_file(path / ".zarray", json);
    }
    return ZarrArray(path, std::move(meta), std::move(codec), std::move(registry));
}

ZarrArray ZarrArray::open(std::shared_ptr<Store> store, const std::string& array_key,
                          Codec codec) {
    // Build store key without leading "/" when array_key is empty.
    auto store_key = [&](const std::string& name) -> std::string {
        return array_key.empty() ? name : array_key + "/" + name;
    };
    ZarrMetadata meta;
    if (store->exists(store_key("zarr.json"))) {
        auto data = store->get_string(store_key("zarr.json"));
        meta = detail::parse_zarr_json(data);
    } else if (store->exists(store_key(".zarray"))) {
        auto data = store->get_string(store_key(".zarray"));
        meta = detail::parse_zarray(data);
    } else {
        throw std::runtime_error("zarr: no metadata found at store key: " + array_key);
    }
    return ZarrArray(std::move(store), array_key, std::move(meta), std::move(codec));
}

ZarrArray ZarrArray::open(std::shared_ptr<Store> store, const std::string& array_key,
                          CodecRegistry registry) {
    auto store_key = [&](const std::string& name) -> std::string {
        return array_key.empty() ? name : array_key + "/" + name;
    };
    ZarrMetadata meta;
    ZarrVersion version = ZarrVersion::v2;
    if (store->exists(store_key("zarr.json"))) {
        auto data = store->get_string(store_key("zarr.json"));
        meta = detail::parse_zarr_json(data);
        version = ZarrVersion::v3;
    } else if (store->exists(store_key(".zarray"))) {
        auto data = store->get_string(store_key(".zarray"));
        meta = detail::parse_zarray(data);
        version = ZarrVersion::v2;
    } else {
        throw std::runtime_error("zarr: no metadata found at store key: " + array_key);
    }

    Codec codec;
    if (version == ZarrVersion::v2 && !meta.compressor_id.empty()) {
        auto it = registry.find(meta.compressor_id);
        if (it != registry.end()) codec = it->second;
    } else if (version == ZarrVersion::v3) {
        auto scan = [&](const std::vector<ZarrCodecConfig>& cs) {
            for (const auto& cc : cs) {
                if (cc.name != "bytes" && cc.name != "transpose"
                    && cc.name != "sharding_indexed") {
                    auto it = registry.find(cc.name);
                    if (it != registry.end()) { codec = it->second; return true; }
                }
            }
            return false;
        };
        if (meta.shard_config) scan(meta.shard_config->sub_codecs);
        if (!codec.decompress) scan(meta.codecs);
    }

    return ZarrArray(std::move(store), array_key, std::move(meta), std::move(codec),
                     std::move(registry));
}

ZarrArray ZarrArray::open_with_metadata(const std::filesystem::path& path,
                                        ZarrMetadata meta, Codec codec) {
    return ZarrArray(path, std::move(meta), std::move(codec));
}

// ---------------------------------------------------------------------------
// ZarrArray -- chunk I/O
// ---------------------------------------------------------------------------

std::optional<std::vector<std::byte>>
ZarrArray::read_chunk(std::span<const std::size_t> chunk_indices) const {
    if (is_sharded()) {
        auto raw = read_inner_chunk_from_shard(chunk_indices);
        if (!raw) return std::nullopt;
        if (codec_.decompress && needs_decompression()) {
            return codec_.decompress(*raw, meta_.sub_chunk_byte_size());
        }
        return raw;
    }

    auto raw = read_chunk_raw(chunk_indices);
    if (!raw) return std::nullopt;

    auto data = std::move(*raw);

    // Decompress.
    if (codec_.decompress && needs_decompression()) {
        data = codec_.decompress(data, meta_.chunk_byte_size());
    }

    // Apply v2 filter decoding (in reverse order).
    if (meta_.version == ZarrVersion::v2) {
        for (auto it = meta_.filters.rbegin(); it != meta_.filters.rend(); ++it)
            data = it->decode(data);
    }

    // Byte-swap if the stored byte order differs from native.
    if (needs_byteswap()) {
        detail::byteswap_inplace(data, dtype_size(meta_.dtype));
    }

    return data;
}

std::optional<std::vector<std::byte>>
ZarrArray::read_chunk_encoded(std::span<const std::size_t> chunk_indices) const {
    if (is_sharded())
        return read_inner_chunk_from_shard(chunk_indices);
    return read_chunk_raw(chunk_indices);
}

std::vector<std::byte>
ZarrArray::decode_chunk_payload(std::span<const std::byte> payload) const {
    std::vector<std::byte> data(payload.begin(), payload.end());
    const std::size_t expected = is_sharded()
        ? meta_.sub_chunk_byte_size()
        : meta_.chunk_byte_size();

    if (codec_.decompress && needs_decompression()) {
        data = codec_.decompress(data, expected);
    }

    if (meta_.version == ZarrVersion::v2) {
        for (auto it = meta_.filters.rbegin(); it != meta_.filters.rend(); ++it)
            data = it->decode(data);
    }

    if (needs_byteswap()) {
        detail::byteswap_inplace(data, dtype_size(meta_.dtype));
    }

    return data;
}

bool ZarrArray::stores_chunks_with_codec(std::string_view codec_name) const noexcept {
    auto has_codec = [codec_name](const std::vector<ZarrCodecConfig>& codecs) {
        for (const auto& codec : codecs) {
            if (codec.name == codec_name)
                return true;
        }
        return false;
    };
    if (meta_.version == ZarrVersion::v2)
        return meta_.compressor_id == codec_name;
    if (meta_.shard_config && has_codec(meta_.shard_config->sub_codecs))
        return true;
    return has_codec(meta_.codecs);
}

bool ZarrArray::read_chunk_into(std::span<const std::size_t> chunk_indices,
                                std::span<std::byte> output) const {
    const std::size_t expected = meta_.sub_chunk_byte_size();
    if (output.size() < expected) {
        throw std::runtime_error("zarr: read_chunk_into output buffer too small");
    }
    auto out = output.subspan(0, expected);

    std::optional<std::vector<std::byte>> raw_opt;
    if (is_sharded()) {
        raw_opt = read_inner_chunk_from_shard(chunk_indices);
    } else {
        raw_opt = read_chunk_raw(chunk_indices);
    }
    if (!raw_opt) return false;

    const bool has_v2_filters =
        (meta_.version == ZarrVersion::v2 && !meta_.filters.empty());
    const bool needs_decode = needs_decompression();

    if (needs_decode && codec_.decompress_into && !has_v2_filters && !needs_byteswap()) {
        codec_.decompress_into(*raw_opt, out);
        return true;
    }

    std::vector<std::byte> data;
    if (needs_decode) {
        if (!codec_.decompress) return false;
        data = codec_.decompress(*raw_opt, expected);
    } else {
        data = std::move(*raw_opt);
    }

    if (has_v2_filters) {
        for (auto it = meta_.filters.rbegin(); it != meta_.filters.rend(); ++it)
            data = it->decode(data);
    }
    if (needs_byteswap()) {
        detail::byteswap_inplace(data, dtype_size(meta_.dtype));
    }
    if (data.size() < expected) return false;
    std::memcpy(out.data(), data.data(), expected);
    return true;
}

void ZarrArray::write_chunk(std::span<const std::size_t> chunk_indices,
                            std::span<const std::byte> data) {
    // Apply v2 filters (forward order).
    std::vector<std::byte> buf;
    std::span<const std::byte> write_data = data;
    if (meta_.version == ZarrVersion::v2 && !meta_.filters.empty()) {
        buf.assign(data.begin(), data.end());
        for (const auto& f : meta_.filters) {
            buf = f.encode(buf);
        }
        write_data = buf;
    }

    // Compress.
    std::vector<std::byte> compressed;
    if (codec_.compress && needs_compression()) {
        compressed = codec_.compress(write_data);
        write_data = compressed;
    }

    write_chunk_raw(chunk_indices, write_data);
}

bool ZarrArray::chunk_exists(std::span<const std::size_t> chunk_indices) const {
    auto key = chunk_key(chunk_indices);
    if (store_)
        return store_->exists(key);
    return std::filesystem::exists(root_ / key);
}

std::string ZarrArray::chunk_key(std::span<const std::size_t> chunk_indices) const {
    if (meta_.version == ZarrVersion::v3)
        return chunk_key_v3(chunk_indices);
    return chunk_key_v2(chunk_indices);
}

std::string ZarrArray::chunk_store_key(std::span<const std::size_t> chunk_indices) const {
    auto key = chunk_key(chunk_indices);
    if (array_key_.empty())
        return key;
    return array_key_ + "/" + key;
}

bool ZarrArray::direct_chunk_payload_is_decoded_bytes() const noexcept {
    if (is_sharded() || needs_compression() || needs_byteswap() || !meta_.filters.empty())
        return false;
    if (meta_.version == ZarrVersion::v2)
        return true;
    for (const auto& codec : meta_.codecs) {
        if (codec.name != "bytes")
            return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// ZarrArray -- shard I/O
// ---------------------------------------------------------------------------

std::optional<std::vector<std::byte>>
ZarrArray::read_inner_chunk(std::span<const std::size_t> shard_indices,
                            std::span<const std::size_t> inner_indices) const {
    if (!is_sharded())
        throw std::runtime_error("zarr: not a sharded array");

    auto shard_key = chunk_key(shard_indices);
    auto shard_data = read_raw(shard_key);
    if (!shard_data) return std::nullopt;
    return extract_inner_chunk(*shard_data, inner_indices);
}

void ZarrArray::write_shard(std::span<const std::size_t> shard_indices,
                            std::span<const std::optional<std::vector<std::byte>>> inner_chunks) {
    if (!is_sharded())
        throw std::runtime_error("zarr: not a sharded array");

    const auto& sc = *meta_.shard_config;
    const auto n_inner = meta_.total_sub_chunks_per_shard();

    if (inner_chunks.size() != n_inner)
        throw std::runtime_error("zarr: wrong number of inner chunks for shard");

    // Build shard data: [index][chunk0 padded to 4k][chunk1 padded to 4k]...
    std::vector<std::byte> shard_data;
    detail::ShardIndex index;
    index.entries.resize(n_inner);

    // Index always at start — reserve space for it.
    const std::size_t index_size = n_inner * 16;
    shard_data.resize(index_size);

    auto pad_to_align = [&]() {
        auto n = shard_data.size();
        auto aligned = (n + kShardChunkAlign - 1) & ~(kShardChunkAlign - 1);
        if (aligned > n) shard_data.resize(aligned, std::byte{0});
    };
    pad_to_align();   // ensures first chunk lands on 4k boundary

    for (std::size_t i = 0; i < n_inner; ++i) {
        if (!inner_chunks[i]) {
            index.entries[i].offset = ~std::uint64_t(0);
            index.entries[i].nbytes  = ~std::uint64_t(0);
            continue;
        }

        const auto& chunk_data = *inner_chunks[i];
        std::span<const std::byte> write_data = chunk_data;

        std::vector<std::byte> compressed;
        if (codec_.compress && needs_compression()) {
            compressed = codec_.compress(write_data);
            write_data = compressed;
        }

        index.entries[i].offset = shard_data.size();
        index.entries[i].nbytes  = write_data.size();
        shard_data.insert(shard_data.end(), write_data.begin(), write_data.end());
        pad_to_align();
    }

    // Write index at start.
    auto index_bytes = index.serialize();
    std::memcpy(shard_data.data(), index_bytes.data(), index_size);

    // Write shard file.
    write_chunk_raw(shard_indices, shard_data);
}

void ZarrArray::write_inner_chunk_to_shard(std::span<const std::size_t> chunk_indices,
                                           std::span<const std::byte> data) {
    if (!is_sharded())
        throw std::runtime_error("zarr: not a sharded array");

    const auto ndim = meta_.ndim();
    const auto n_inner = meta_.total_sub_chunks_per_shard();
    const std::size_t index_size = n_inner * 16;

    std::vector<std::size_t> shard_idx(ndim);
    std::vector<std::size_t> inner_idx(ndim);
    for (std::size_t d = 0; d < ndim; ++d) {
        auto ips = meta_.sub_chunks_per_shard(d);
        shard_idx[d] = chunk_indices[d] / ips;
        inner_idx[d] = chunk_indices[d] % ips;
    }

    std::size_t linear = 0;
    std::size_t stride = 1;
    for (std::size_t d = ndim; d-- > 0;) {
        linear += inner_idx[d] * stride;
        stride *= meta_.sub_chunks_per_shard(d);
    }

    auto key = chunk_key(shard_idx);
    auto p = root_ / key;
    std::filesystem::create_directories(p.parent_path());

    // Lock to prevent concurrent writes tearing this shard file.
    // Striped so writers to different shards run concurrently.
    //
    // Must cover the exists-check-and-create pair: without the lock
    // two writers racing on a fresh shard would both see !exists,
    // both truncate with an empty index, and the second writer would
    // overwrite the first's already-committed index entry.
    std::lock_guard lock(shard_mutex_for(p));

    if (!std::filesystem::exists(p)) {
        std::ofstream create(p, std::ios::binary);
        // Write empty index: all entries = (0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF)
        std::vector<std::byte> empty_index(index_size);
        std::memset(empty_index.data(), 0xFF, index_size);
        create.write(reinterpret_cast<const char*>(empty_index.data()),
                     static_cast<std::streamsize>(index_size));
    }

    // Open for random read/write
    std::fstream f(p, std::ios::binary | std::ios::in | std::ios::out);
    if (!f) return;

    // 1. Seek to EOF, round up to next 4k boundary, append chunk data.
    //    Padding bytes (if any) are left uninitialised — ext4 zero-fills
    //    them on sparse extension, and readers never look at them.
    f.seekp(0, std::ios::end);
    auto eof_offset = static_cast<std::uint64_t>(f.tellp());
    auto chunk_offset = (eof_offset + kShardChunkAlign - 1)
                      & ~(kShardChunkAlign - 1);
    if (chunk_offset != eof_offset) {
        f.seekp(static_cast<std::streamoff>(chunk_offset));
    }
    f.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size()));

    // 2. Seek to index entry, write 16 bytes (offset + nbytes)
    auto nbytes = static_cast<std::uint64_t>(data.size());
    f.seekp(static_cast<std::streamoff>(linear * 16));
    f.write(reinterpret_cast<const char*>(&chunk_offset), 8);
    f.write(reinterpret_cast<const char*>(&nbytes), 8);
    f.flush();
}

bool ZarrArray::inner_chunk_exists(std::span<const std::size_t> chunk_indices) const {
    if (!is_sharded()) return false;
    const auto ndim = meta_.ndim();

    std::vector<std::size_t> shard_idx(ndim);
    std::vector<std::size_t> inner_idx(ndim);
    for (std::size_t d = 0; d < ndim; ++d) {
        auto ips = meta_.sub_chunks_per_shard(d);
        shard_idx[d] = chunk_indices[d] / ips;
        inner_idx[d] = chunk_indices[d] % ips;
    }

    std::size_t linear = 0;
    std::size_t stride = 1;
    for (std::size_t d = ndim; d-- > 0;) {
        linear += inner_idx[d] * stride;
        stride *= meta_.sub_chunks_per_shard(d);
    }

    auto p = root_ / chunk_key(shard_idx);
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;

    f.seekg(static_cast<std::streamoff>(linear * 16));
    std::uint64_t offset = 0, nbytes = 0;
    f.read(reinterpret_cast<char*>(&offset), 8);
    f.read(reinterpret_cast<char*>(&nbytes), 8);
    if (!f) return false;
    // Not present: (0xFF..FF, 0xFF..FF). Empty/zero: (0xFF..FE, 0).
    if (offset == ~std::uint64_t(0) && nbytes == ~std::uint64_t(0)) return false;
    if (offset == (~std::uint64_t(0) - 1) && nbytes == 0) return false;
    return true;
}

bool ZarrArray::inner_chunk_is_empty(std::span<const std::size_t> chunk_indices) const {
    if (!is_sharded()) return false;
    const auto ndim = meta_.ndim();

    std::vector<std::size_t> shard_idx(ndim);
    std::vector<std::size_t> inner_idx(ndim);
    for (std::size_t d = 0; d < ndim; ++d) {
        auto ips = meta_.sub_chunks_per_shard(d);
        shard_idx[d] = chunk_indices[d] / ips;
        inner_idx[d] = chunk_indices[d] % ips;
    }

    std::size_t linear = 0;
    std::size_t stride = 1;
    for (std::size_t d = ndim; d-- > 0;) {
        linear += inner_idx[d] * stride;
        stride *= meta_.sub_chunks_per_shard(d);
    }

    auto p = root_ / chunk_key(shard_idx);
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;

    f.seekg(static_cast<std::streamoff>(linear * 16));
    std::uint64_t offset = 0, nbytes = 0;
    f.read(reinterpret_cast<char*>(&offset), 8);
    f.read(reinterpret_cast<char*>(&nbytes), 8);
    if (!f) return false;
    return (offset == (~std::uint64_t(0) - 1) && nbytes == 0);
}

void ZarrArray::mark_inner_chunk_empty(std::span<const std::size_t> chunk_indices) {
    if (!is_sharded())
        throw std::runtime_error("zarr: not a sharded array");

    const auto ndim = meta_.ndim();
    const auto n_inner = meta_.total_sub_chunks_per_shard();
    const std::size_t index_size = n_inner * 16;

    std::vector<std::size_t> shard_idx(ndim);
    std::vector<std::size_t> inner_idx(ndim);
    for (std::size_t d = 0; d < ndim; ++d) {
        auto ips = meta_.sub_chunks_per_shard(d);
        shard_idx[d] = chunk_indices[d] / ips;
        inner_idx[d] = chunk_indices[d] % ips;
    }

    std::size_t linear = 0;
    std::size_t stride = 1;
    for (std::size_t d = ndim; d-- > 0;) {
        linear += inner_idx[d] * stride;
        stride *= meta_.sub_chunks_per_shard(d);
    }

    auto key = chunk_key(shard_idx);
    auto p = root_ / key;
    std::filesystem::create_directories(p.parent_path());

    if (!std::filesystem::exists(p)) {
        std::ofstream create(p, std::ios::binary);
        std::vector<std::byte> empty_index(index_size);
        std::memset(empty_index.data(), 0xFF, index_size);
        create.write(reinterpret_cast<const char*>(empty_index.data()),
                     static_cast<std::streamsize>(index_size));
    }

    std::lock_guard lock(shard_mutex_for(p));
    std::fstream f(p, std::ios::binary | std::ios::in | std::ios::out);
    if (!f) return;

    // Write empty sentinel: (0xFF..FE, 0)
    std::uint64_t sentinel_offset = ~std::uint64_t(0) - 1;
    std::uint64_t sentinel_nbytes = 0;
    f.seekp(static_cast<std::streamoff>(linear * 16));
    f.write(reinterpret_cast<const char*>(&sentinel_offset), 8);
    f.write(reinterpret_cast<const char*>(&sentinel_nbytes), 8);
    f.flush();
}

void ZarrArray::write_empty_shard(std::span<const std::size_t> shard_indices) {
    if (!is_sharded())
        throw std::runtime_error("zarr: not a sharded array");
    const auto n_inner = meta_.total_sub_chunks_per_shard();

    auto key = chunk_key(std::vector<std::size_t>(shard_indices.begin(), shard_indices.end()));
    auto p = root_ / key;
    std::filesystem::create_directories(p.parent_path());

    std::vector<std::uint64_t> index(n_inner * 2);
    const std::uint64_t sentinel_offset = ~std::uint64_t(0) - 1;
    const std::uint64_t sentinel_nbytes = 0;
    for (std::size_t i = 0; i < n_inner; ++i) {
        index[i * 2]     = sentinel_offset;
        index[i * 2 + 1] = sentinel_nbytes;
    }

    std::lock_guard lock(shard_mutex_for(p));
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    if (!f) return;
    f.write(reinterpret_cast<const char*>(index.data()),
            static_cast<std::streamsize>(index.size() * 8));
}

// ---------------------------------------------------------------------------
// ZarrArray -- internal helpers
// ---------------------------------------------------------------------------

bool ZarrArray::needs_compression() const noexcept {
    if (meta_.version == ZarrVersion::v2)
        return !meta_.compressor_id.empty();
    // v3: check for bytes-to-bytes codecs in pipeline.
    auto has_bb = [](const std::vector<ZarrCodecConfig>& cs) {
        for (const auto& cc : cs)
            if (cc.name != "bytes" && cc.name != "transpose"
                && cc.name != "sharding_indexed")
                return true;
        return false;
    };
    if (meta_.shard_config && has_bb(meta_.shard_config->sub_codecs)) return true;
    return has_bb(meta_.codecs);
}

bool ZarrArray::needs_decompression() const noexcept {
    return needs_compression();
}

std::string ZarrArray::chunk_key_v2(std::span<const std::size_t> idx) const {
    std::string name;
    for (std::size_t i = 0; i < idx.size(); ++i) {
        if (i) name += meta_.dimension_separator;
        name += std::to_string(idx[i]);
    }
    return name;
}

std::string ZarrArray::chunk_key_v3(std::span<const std::size_t> idx) const {
    std::string sep = meta_.v3_separator();
    std::string name = "c";
    for (std::size_t i = 0; i < idx.size(); ++i) {
        name += sep;
        name += std::to_string(idx[i]);
    }
    return name;
}

std::optional<std::vector<std::byte>>
ZarrArray::read_raw(const std::string& key) const {
    if (store_) {
        auto full_key = array_key_.empty() ? key : array_key_ + "/" + key;
        return store_->get_if_exists(full_key);
    }
    auto p = root_ / key;
    if (!std::filesystem::exists(p)) return std::nullopt;
    return detail::read_file_bytes(p);
}

void ZarrArray::write_chunk_raw(std::span<const std::size_t> idx,
                                std::span<const std::byte> data) {
    auto key = chunk_key(idx);
    if (store_) {
        auto full_key = array_key_.empty() ? key : array_key_ + "/" + key;
        store_->set(full_key, data);
        return;
    }
    auto p = root_ / key;
    std::filesystem::create_directories(p.parent_path());
    detail::write_file_bytes(p, data);
}

std::optional<std::vector<std::byte>>
ZarrArray::read_inner_chunk_from_shard(std::span<const std::size_t> chunk_indices) const {
    if (!is_sharded()) return std::nullopt;
    const auto ndim = meta_.ndim();

    std::vector<std::size_t> shard_idx(ndim);
    std::vector<std::size_t> inner_idx(ndim);
    for (std::size_t d = 0; d < ndim; ++d) {
        auto ips = meta_.sub_chunks_per_shard(d);
        shard_idx[d] = chunk_indices[d] / ips;
        inner_idx[d] = chunk_indices[d] % ips;
    }

    std::size_t linear = 0;
    std::size_t stride = 1;
    for (std::size_t d = ndim; d-- > 0;) {
        linear += inner_idx[d] * stride;
        stride *= meta_.sub_chunks_per_shard(d);
    }
    if (linear > std::numeric_limits<std::size_t>::max() / 16)
        return std::nullopt;
    const std::size_t index_offset = linear * 16;
    auto is_missing_or_empty = [](std::uint64_t offset, std::uint64_t nbytes) {
        return (offset == ~std::uint64_t(0) && nbytes == ~std::uint64_t(0)) ||
               (offset == ~std::uint64_t(0) - 1 && nbytes == 0) ||
               nbytes == 0;
    };

    auto key = chunk_key(shard_idx);
    if (store_) {
        auto full_key = array_key_.empty() ? key : array_key_ + "/" + key;
        auto entry = store_->get_partial(full_key, index_offset, 16);
        if (!entry || entry->size() < 16) return std::nullopt;

        std::uint64_t offset = 0, nbytes = 0;
        std::memcpy(&offset, entry->data(), 8);
        std::memcpy(&nbytes, entry->data() + 8, 8);
        if (is_missing_or_empty(offset, nbytes)) return std::nullopt;
        if (offset > std::numeric_limits<std::size_t>::max() ||
            nbytes > std::numeric_limits<std::size_t>::max())
            return std::nullopt;

        auto data = store_->get_partial(full_key,
                                        static_cast<std::size_t>(offset),
                                        static_cast<std::size_t>(nbytes));
        if (!data || data->size() != static_cast<std::size_t>(nbytes))
            return std::nullopt;
        return data;
    }

    auto p = root_ / key;
    // Lock to prevent reading while another thread is writing the
    // same shard (striped — reads against other shards don't block).
    std::lock_guard lock(shard_mutex_for(p));
    std::ifstream f(p, std::ios::binary);
    if (!f) return std::nullopt;

    // Read 16-byte index entry at position linear*16
    if (index_offset > static_cast<std::size_t>(std::numeric_limits<std::streamoff>::max()))
        return std::nullopt;
    f.seekg(static_cast<std::streamoff>(index_offset));
    std::uint64_t offset = 0, nbytes = 0;
    f.read(reinterpret_cast<char*>(&offset), 8);
    f.read(reinterpret_cast<char*>(&nbytes), 8);
    if (!f) return std::nullopt;
    if (is_missing_or_empty(offset, nbytes)) return std::nullopt;
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max()) ||
        nbytes > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()))
        return std::nullopt;

    // Read chunk data
    f.seekg(static_cast<std::streamoff>(offset));
    std::vector<std::byte> data(nbytes);
    f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(nbytes));
    if (!f) return std::nullopt;
    return data;
}

std::optional<ShardBytes>
ZarrArray::read_whole_shard(std::span<const std::size_t> chunk_indices) const {
    if (!is_sharded()) return std::nullopt;
    const auto ndim = meta_.ndim();
    std::vector<std::size_t> shard_idx(ndim);
    for (std::size_t d = 0; d < ndim; ++d) {
        auto ips = meta_.sub_chunks_per_shard(d);
        shard_idx[d] = chunk_indices[d] / ips;
    }
    auto key = chunk_key(shard_idx);
    auto p = root_ / key;
    std::lock_guard lock(shard_mutex_for(p));
#if !defined(_WIN32)
    int fd = ::open(p.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return std::nullopt;
    struct stat st;
    if (::fstat(fd, &st) < 0 || st.st_size <= 0) {
        ::close(fd);
        return std::nullopt;
    }
    const std::size_t sz = static_cast<std::size_t>(st.st_size);
    void* ptr = ::mmap(nullptr, sz, PROT_READ, MAP_PRIVATE, fd, 0);
    // The mapping survives the fd close — no fd leak from a long-lived
    // shard-cache entry.
    ::close(fd);
    if (ptr == MAP_FAILED) return std::nullopt;
    // MADV_RANDOM: we parse the trailing index then hop to a specific
    // inner-chunk offset. Sequential readahead would prefetch pages we
    // never touch.
    ::madvise(ptr, sz, MADV_RANDOM);
    return ShardBytes::from_mmap(ptr, sz);
#else
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) return std::nullopt;
    auto size = static_cast<std::streamsize>(f.tellg());
    if (size <= 0) return std::nullopt;
    std::vector<std::byte> data(static_cast<std::size_t>(size));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(data.data()), size);
    if (!f) return std::nullopt;
    return ShardBytes::from_vector(std::move(data));
#endif
}

// ---------------------------------------------------------------------------
// open_remote
// ---------------------------------------------------------------------------

ZarrArray open_remote(
    const std::string& url,
    std::optional<AwsAuth> auth,
    ZarrArray::Codec codec,
    const std::string& array_key)
{
    std::string base_url = url;
    std::shared_ptr<Store> store;

    if (is_s3_url(url)) {
        auto parsed = parse_s3_url(url);
        if (!parsed)
            throw std::runtime_error("open_remote: invalid S3 URL: " + url);

        base_url = s3_to_https(*parsed);

        // Resolve auth: explicit > load (SSO/files/env)
        AwsAuth aws = auth.value_or(AwsAuth::load());

        // If the URL included a region, prefer that over env/explicit
        if (!parsed->region.empty())
            aws.region = parsed->region;

        store = std::make_shared<HttpStore>(base_url, std::move(aws));
    } else {
        if (auth) {
            store = std::make_shared<HttpStore>(base_url, std::move(*auth));
        } else {
            store = std::make_shared<HttpStore>(base_url);
        }
    }

    return ZarrArray::open(std::move(store), array_key, std::move(codec));
}

// ---------------------------------------------------------------------------
// Consolidated metadata + pyramid helpers
// ---------------------------------------------------------------------------

ZarrArray open_from_consolidated(
    const std::filesystem::path& root,
    const std::string& array_path,
    const detail::ConsolidatedMetadata& cm,
    ZarrArray::Codec codec) {
    auto it = cm.arrays.find(array_path);
    if (it == cm.arrays.end())
        throw std::runtime_error("zarr: array not found in consolidated metadata: " + array_path);
    return ZarrArray::open_with_metadata(root / array_path, it->second, std::move(codec));
}

std::size_t count_pyramid_levels(const std::filesystem::path& root) {
    std::size_t level = 0;
    while (std::filesystem::is_directory(root / std::to_string(level)))
        ++level;
    return level;
}

std::vector<ZarrArray> open_pyramid(
    const std::filesystem::path& root, ZarrArray::Codec codec) {
    std::vector<ZarrArray> levels;
    auto n = count_pyramid_levels(root);
    levels.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        levels.push_back(ZarrArray::open(root / std::to_string(i), codec));
    return levels;
}

// ---------------------------------------------------------------------------
// MultiscaleMetadata::voxel_size
// ---------------------------------------------------------------------------

std::array<double, 3> MultiscaleMetadata::voxel_size(std::size_t level) const {
    if (level >= datasets.size())
        throw std::runtime_error("zarr: OME level out of range");

    std::array<double, 3> vs{1.0, 1.0, 1.0};
    const auto& ds = datasets[level];

    const ScaleTransform* st = nullptr;
    for (const auto& t : ds.transforms) {
        if (auto* p = std::get_if<ScaleTransform>(&t)) {
            st = p;
            break;
        }
    }
    if (!st) return vs;

    auto xi = axis_index("x");
    auto yi = axis_index("y");
    auto zi = axis_index("z");
    if (xi && *xi < st->scale.size()) vs[0] = st->scale[*xi];
    if (yi && *yi < st->scale.size()) vs[1] = st->scale[*yi];
    if (zi && *zi < st->scale.size()) vs[2] = st->scale[*zi];
    return vs;
}

// ---------------------------------------------------------------------------
// ome_detail axis-type helpers
// ---------------------------------------------------------------------------

namespace ome_detail {

AxisType parse_axis_type(std::string_view s) noexcept {
    if (s == "space") return AxisType::space;
    if (s == "time")  return AxisType::time;
    if (s == "channel") return AxisType::channel;
    return AxisType::custom;
}

std::string_view axis_type_string(AxisType t) noexcept {
    switch (t) {
        case AxisType::space:   return "space";
        case AxisType::time:    return "time";
        case AxisType::channel: return "channel";
        case AxisType::custom:  return "custom";
    }
    return "custom";
}

} // namespace ome_detail

// ---------------------------------------------------------------------------
// Pyramid generation helpers
// ---------------------------------------------------------------------------

std::vector<double> compute_downsample_factors(
    const MultiscaleMetadata& meta, std::size_t from_level, std::size_t to_level) {
    if (from_level >= meta.datasets.size() || to_level >= meta.datasets.size())
        throw std::runtime_error("zarr: level index out of range");

    auto get_scale = [&](std::size_t level) -> const std::vector<double>& {
        for (const auto& t : meta.datasets[level].transforms) {
            if (auto* st = std::get_if<ScaleTransform>(&t))
                return st->scale;
        }
        throw std::runtime_error(
            "zarr: no scale transform at level " + std::to_string(level));
    };

    const auto& from_scale = get_scale(from_level);
    const auto& to_scale = get_scale(to_level);

    std::size_t ndim = std::min(from_scale.size(), to_scale.size());
    std::vector<double> factors(ndim);
    for (std::size_t i = 0; i < ndim; ++i) {
        factors[i] = (from_scale[i] != 0.0) ? (to_scale[i] / from_scale[i]) : 1.0;
    }
    return factors;
}

MultiscaleMetadata make_standard_multiscale(
    std::string_view name,
    std::vector<std::size_t> full_shape,
    std::size_t num_levels,
    std::vector<Axis> axes,
    std::array<double, 3> voxel_size) {

    (void)full_shape;

    MultiscaleMetadata meta;
    meta.version = "0.4";
    meta.name = std::string(name);
    meta.type = "gaussian";
    meta.axes = std::move(axes);

    auto xi = meta.axis_index("x");
    auto yi = meta.axis_index("y");
    auto zi = meta.axis_index("z");

    for (std::size_t lvl = 0; lvl < num_levels; ++lvl) {
        MultiscaleDataset ds;
        ds.path = std::to_string(lvl);

        double factor = static_cast<double>(std::size_t{1} << lvl);
        std::vector<double> scale(meta.axes.size(), 1.0);
        if (xi && *xi < scale.size()) scale[*xi] = voxel_size[0] * factor;
        if (yi && *yi < scale.size()) scale[*yi] = voxel_size[1] * factor;
        if (zi && *zi < scale.size()) scale[*zi] = voxel_size[2] * factor;
        ds.transforms.emplace_back(ScaleTransform{std::move(scale)});

        meta.datasets.push_back(std::move(ds));
    }

    return meta;
}

} // namespace utils
