// Objects below a local directory or an s3:// prefix, for writing.
#include <fstream>
#include <unistd.h>
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
        return access.url(name.empty() ? location : joinLocation(location, name));
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

void Store::writeFile(const std::string& name, const fs::path& file) const
{
    const auto bytes = fs::file_size(file);
    if (!impl_->remote) {
        const auto target = fs::path(impl_->location) / name;
        fs::create_directories(target.parent_path());
        auto temporary = target;
        temporary += "." + std::to_string(::getpid()) + ".tmp";
        fs::copy_file(file, temporary, fs::copy_options::overwrite_existing);
        fs::rename(temporary, target);
        return;
    }
    impl_->needWritable();
    constexpr std::uint64_t kPartBytes = 64ULL << 20;
    std::ifstream in(file, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + file.string());
    std::vector<char> buffer(std::size_t(std::min<std::uint64_t>(std::max(kPartBytes, bytes / 9000 + 1), std::max<std::uint64_t>(bytes, 1))));
    if (bytes <= kPartBytes) {
        in.read(buffer.data(), std::streamsize(bytes));
        impl_->signedRequest([&] { impl_->http->put(impl_->url(name), buffer.data(), std::size_t(bytes)); });
        return;
    }
    const auto url = impl_->url(name);
    const auto started = impl_->signedRequest([&] { return impl_->http->post(url + "?uploads=", "", 0); });
    const auto open = started.find("<UploadId>"), close = started.find("</UploadId>");
    if (open == std::string::npos || close == std::string::npos) throw std::runtime_error("starting an upload of " + url + ": no UploadId");
    const auto id = started.substr(open + 10, close - open - 10);
    try {
        std::string done = "<CompleteMultipartUpload>";
        std::uint64_t sent = 0;
        for (int part = 1; sent < bytes; ++part) {
            const auto size = std::size_t(std::min<std::uint64_t>(buffer.size(), bytes - sent));
            if (!in.read(buffer.data(), std::streamsize(size))) throw std::runtime_error("cannot read " + file.string());
            const auto partUrl = url + "?partNumber=" + std::to_string(part) + "&uploadId=" + id;
            const auto etag = impl_->signedRequest([&] { return impl_->http->put_part(partUrl, buffer.data(), size); });
            done += "<Part><PartNumber>" + std::to_string(part) + "</PartNumber><ETag>" + etag + "</ETag></Part>";
            sent += size;
        }
        done += "</CompleteMultipartUpload>";
        const auto answer = impl_->signedRequest([&] { return impl_->http->post(url + "?uploadId=" + id, done.data(), done.size()); });
        // S3 may answer 200 and still report a failure in the body.
        if (answer.find("<Error>") != std::string::npos) throw std::runtime_error("completing an upload of " + url + ": " + answer);
    } catch (...) {
        try {
            impl_->http->remove(url + "?uploadId=" + id);
        } catch (...) {
        }
        throw;
    }
}

bool Store::readFile(const std::string& name, const fs::path& file) const
{
    if (file.has_parent_path()) fs::create_directories(file.parent_path());
    auto temporary = file;
    temporary += "." + std::to_string(::getpid()) + ".tmp";
    const auto finish = [&](bool found) {
        if (found) fs::rename(temporary, file);
        else fs::remove(temporary);
        return found;
    };
    if (!impl_->remote) {
        const auto source = fs::path(impl_->location) / name;
        if (!fs::exists(source)) return false;
        fs::copy_file(source, temporary, fs::copy_options::overwrite_existing);
        return finish(true);
    }
    if (!impl_->http) {
        const auto data = read(name);
        if (!data) return false;
        atomicWrite(file, *data);
        return true;
    }
    std::ofstream out;
    const auto response = impl_->signedRequest([&] {
        return impl_->http->get_stream(impl_->url(name), [&] {
            out.close();
            out.open(temporary, std::ios::binary | std::ios::trunc);
            if (!out) throw std::runtime_error("cannot write " + temporary.string());
        }, [&](const char* data, std::size_t n) { out.write(data, std::streamsize(n)); });
    });
    out.close();
    if (!out) {
        std::error_code ignored;
        fs::remove(temporary, ignored);
        throw std::runtime_error("cannot write " + temporary.string());
    }
    return finish(response.status == 200);
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

std::vector<s3::Object> Store::list(const std::string& keyPrefix) const
{
    std::vector<s3::Object> objects;
    if (!impl_->remote) {
        std::error_code ec;
        for (fs::recursive_directory_iterator it(impl_->location, ec), end; !ec && it != end; it.increment(ec))
            if (it->is_regular_file(ec)) {
                auto key = it->path().lexically_relative(impl_->location).generic_string();
                if (key.starts_with(keyPrefix))
                    objects.push_back({std::move(key), it->file_size(ec), {}, {}});
            }
        std::sort(objects.begin(), objects.end(), [](const auto& a, const auto& b) { return a.key < b.key; });
        return objects;
    }
    return impl_->signedRequest([&] { return listObjects(impl_->access, impl_->location, keyPrefix); });
}

Listing Store::listPage(const std::string& keyPrefix, const std::string& continuationToken, int maxKeys) const
{
    if (impl_->remote)
        return impl_->signedRequest([&] { return zarr::listPage(impl_->access, impl_->location, keyPrefix, continuationToken, maxKeys); });
    if (maxKeys < 1) throw std::invalid_argument("listing: max-keys must be positive");
    // The directory named by the prefix's directory part, entries starting with its name part.
    const auto slash = keyPrefix.rfind('/');
    const std::string directory = slash == std::string::npos ? std::string() : keyPrefix.substr(0, slash + 1);
    const std::string namePrefix = slash == std::string::npos ? keyPrefix : keyPrefix.substr(slash + 1);
    struct Item { std::string key; bool directory; std::uint64_t size; };
    std::vector<Item> items;
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(fs::path(impl_->location) / directory, ec)) {
        const auto name = entry.path().filename().string();
        if (!name.starts_with(namePrefix)) continue;
        if (entry.is_directory(ec)) items.push_back({directory + name, true, 0});
        else if (entry.is_regular_file(ec)) items.push_back({directory + name, false, entry.file_size(ec)});
    }
    std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) { return a.key < b.key; });
    std::size_t start = 0;
    if (!continuationToken.empty()) start = std::stoull(continuationToken);
    const auto end = std::min(items.size(), start + std::size_t(maxKeys));
    Listing listing;
    for (auto i = start; i < end; ++i) {
        if (items[i].directory) listing.children.push_back(items[i].key);
        else listing.objects.push_back({items[i].key, items[i].size, {}, {}});
    }
    if (end < items.size()) listing.nextContinuationToken = std::to_string(end);
    return listing;
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
