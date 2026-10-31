// Objects below a local directory or an s3:// prefix, for writing.
#include "vc_reader/s3_listing.hpp"
#include "vc_reader/zarr_write.hpp"
#include "vc_reader/native/http.hpp"
#include "vc_reader/native/thread_pool.hpp"

#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

namespace vcr::zarr {

struct Store::Impl {
    std::string location;
    bool remote = false;
    Access access;
    int connections = 16;
    std::shared_ptr<HttpClient> http;
    MetadataReader reader;
    std::mutex mutex;  // guards reader and access across a key reload

    Impl(const fs::path& path, const Access& keys, int count)
        : location(path.string()), remote(isRemote(location)), access(keys), connections(count)
    {
        while (location.size() > 1 && location.back() == '/')
            location.pop_back();
        reader = metadataReader(access, connections);
        if (remote && writable()) {
            HttpOptions options;
            options.connections = connections;
            options.auth = access.auth;
            http = std::make_shared<HttpClient>(options);
        }
    }

    bool writable() const { return location.starts_with("s3://") && access.kind == Access::Kind::S3; }

    std::string url(const std::string& name) const
    {
        return access.url(name.empty() ? location : location + "/" + name);
    }

    void needWritable() const
    {
        if (remote && !writable())
            throw std::invalid_argument("writing to " + location + " needs an s3:// URL and S3 credentials");
    }

    // Temporary keys from the environment are read again once when they stop working.
    template <class F>
    auto signedRequest(F&& request) -> decltype(request())
    {
        try {
            return request();
        } catch (const HttpDenied&) {
            std::lock_guard lock(mutex);
            if (!access.fromEnvironment)
                throw;
            reloadEnvironmentKeys(access);
            if (http)
                http->set_auth(access.auth);
            reader = metadataReader(access, connections);
        }
        return request();
    }

    MetadataReader currentReader()
    {
        std::lock_guard lock(mutex);
        return reader;
    }
};

Store::Store(const fs::path& location, const Access& access, int connections)
    : impl_(std::make_unique<Impl>(location, access, connections))
{}
Store::Store(Store&&) noexcept = default;
Store& Store::operator=(Store&&) noexcept = default;
Store::~Store() = default;

const std::string& Store::location() const { return impl_->location; }
const Access& Store::access() const { return impl_->access; }
bool Store::remote() const { return impl_->remote; }

std::optional<std::string> Store::read(const std::string& name) const
{
    return impl_->signedRequest([&] { return impl_->currentReader()(impl_->location, name); });
}

void Store::write(const std::string& name, std::string_view data) const
{
    if (!impl_->remote) {
        atomicWrite(fs::path(impl_->location) / name, std::string(data));
        return;
    }
    impl_->needWritable();
    impl_->signedRequest([&] { impl_->http->put(impl_->url(name), data.data(), data.size()); });
}

bool Store::exists(const std::string& name) const
{
    if (!impl_->remote) {
        std::error_code ec;
        return fs::exists(fs::path(impl_->location) / name, ec);
    }
    if (!impl_->http) return read(name).has_value();
    return impl_->signedRequest([&] { return impl_->http->head(impl_->url(name)); });
}

bool Store::remove(const std::string& name) const
{
    if (impl_->remote && !exists(name)) {
        impl_->needWritable();
        return false;
    }
    if (impl_->remote) {
        discard(name);
        return true;
    }
    const auto path = fs::path(impl_->location) / name;
    std::error_code ec;
    const bool removed = fs::remove(path, ec);
    if (ec)
        throw std::system_error(ec, "removing " + path.string());
    return removed;
}

void Store::discard(const std::string& name) const
{
    if (!impl_->remote) {
        remove(name);
        return;
    }
    impl_->needWritable();
    impl_->signedRequest([&] { impl_->http->remove(impl_->url(name)); });
}

bool Store::removeAll() const
{
    if (!impl_->remote)
        return fs::remove_all(impl_->location) > 0;
    impl_->needWritable();
    const auto objects = list();
    ThreadPool pool(std::size_t(std::max(impl_->connections - 1, 0)));
    pool.parallel_for(objects.size(), [&](std::size_t i) { discard(objects[i].key); });
    return !objects.empty();
}

std::vector<s3::Object> Store::list() const
{
    std::vector<s3::Object> objects;
    if (!impl_->remote) {
        std::error_code ec;
        for (fs::recursive_directory_iterator it(impl_->location, ec), end; !ec && it != end; it.increment(ec))
            if (it->is_regular_file(ec))
                objects.push_back({it->path().lexically_relative(impl_->location).generic_string(), it->file_size(ec)});
        std::sort(objects.begin(), objects.end(), [](const auto& a, const auto& b) { return a.key < b.key; });
        return objects;
    }
    return impl_->signedRequest([&] { return listObjects(impl_->access, impl_->location); });
}

std::vector<std::string> Store::children() const
{
    std::vector<std::string> names;
    if (!impl_->remote) {
        std::error_code ec;
        for (const auto& child : fs::directory_iterator(impl_->location, ec))
            if (child.is_directory())
                names.push_back(child.path().filename().string());
    } else {
        names = impl_->signedRequest([&] { return listChildren(impl_->access, impl_->location); });
    }
    std::sort(names.begin(), names.end());
    return names;
}

bool Store::isNode() const
{
    for (const auto* name : {".zarray", ".zgroup", ".zattrs", "zarr.json"})
        if (exists(name)) return true;
    return false;
}

bool Store::isArray() const
{
    if (exists(".zarray")) return true;
    const auto text = read("zarr.json");
    if (!text) return false;
    const auto node = Json::parse(*text, nullptr, false);
    return node.is_object() && node.value("node_type", "") == "array";
}

}  // namespace vcr::zarr
