#include "utils/zarr.hpp"

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
            if (auto* cl = json_find(*p, "clevel"); cl && cl->is_number())
                meta.compression_level = cl->get_int();
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

} // namespace utils
