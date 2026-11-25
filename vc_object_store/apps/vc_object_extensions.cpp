// vc_object_extensions: list a store's extension tables, or drop tables of
// extensions that no longer exist.
#include "vc_object_store/objects/ObjectStore.hpp"

#include <algorithm>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace vc::objects;

int main(int argc, char** argv)
try {
    std::filesystem::path path;
    std::string author;
    std::vector<std::string> drop;
    bool list = false, yes = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (++i >= argc) throw std::runtime_error("missing value for " + arg);
            return argv[i];
        };
        if (arg == "--db") path = next();
        else if (arg == "--author") author = next();
        else if (arg == "--list") list = true;
        else if (arg == "--drop") drop.push_back(next());
        else if (arg == "--yes") yes = true;
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: vc_object_extensions --db objects.vcobj --list\n"
                         "       vc_object_extensions --db objects.vcobj --author NAME --drop TABLE [--drop TABLE ...] [--yes]\n"
                         "  --list        print every extension table: name, extension, format, rows\n"
                         "  --drop TABLE  remove the table with all its rows and its registration\n"
                         "  --yes         really drop; without it the tables and row counts are only printed\n"
                         "A drop cannot be undone and is not recorded in the store's history. Close every\n"
                         "program that has the store open first. All named tables are dropped in one commit;\n"
                         "a name the store does not register fails the whole call.\n";
            return 0;
        } else throw std::runtime_error("unknown argument " + arg);
    }
    if (path.empty()) throw std::runtime_error("--db is required");
    if (list == !drop.empty()) throw std::runtime_error("give either --list or --drop");
    if (!std::filesystem::exists(path)) throw std::runtime_error("no such store: " + path.string());
    OpenOptions options;
    options.author = author.empty() ? "vc_object_extensions" : author;
    if (!drop.empty() && author.empty()) throw std::runtime_error("--author is required to drop");
    auto store = ObjectStore::open(path, options);
    const auto rows = [&](const Txn& txn, const std::string& table) {
        std::int64_t count = 0;
        store.scanExtensionRows(txn, table, [&](const ExtensionRow&) { ++count; return true; });
        return count;
    };
    if (list) {
        auto txn = store.read();
        for (const auto& table : store.extensionTables(txn))
            std::cout << table.name << "\t" << table.extension << "\tformat " << table.formatVersion << "\t"
                      << rows(txn, table.name) << " rows\n";
        return 0;
    }
    {
        auto txn = store.read();
        const auto tables = store.extensionTables(txn);
        for (const auto& name : drop) {
            if (std::count(drop.begin(), drop.end(), name) != 1) throw std::runtime_error("table named twice: " + name);
            const auto found = std::find_if(tables.begin(), tables.end(), [&](const auto& t) { return t.name == name; });
            if (found == tables.end()) throw std::runtime_error("the store registers no extension table " + name);
            std::cout << (yes ? "dropping " : "would drop ") << name << " (extension " << found->extension << ", "
                      << rows(txn, name) << " rows)\n";
        }
    }
    if (!yes) {
        std::cout << "nothing was changed; repeat with --yes to drop\n";
        return 0;
    }
    auto txn = store.write();
    for (const auto& name : drop) store.dropExtensionTable(txn, name);
    txn.commit();
    std::cout << "dropped " << drop.size() << " tables\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "vc_object_extensions: " << error.what() << '\n';
    return 1;
}
