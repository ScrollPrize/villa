// Explicit offline schema conversion and history-policy maintenance.
#include "vc_object_store/migration/ConvertStore.hpp"
#include "vc_object_store/objects/StoreConversion.hpp"

#include <iostream>
#include <string>

int main(int argc, char** argv)
{
    try {
        if (argc == 3 && (std::string(argv[1]) == "--enable-history" ||
                          std::string(argv[1]) == "--remove-history-and-disable")) {
            vc::objects::changeHistoryMode(argv[2], std::string(argv[1]) == "--enable-history"
                ? vc::objects::HistoryMode::On : vc::objects::HistoryMode::Off);
            std::cout << "History policy updated atomically\n";
            return 0;
        }
        if (argc < 3 || argc > 5 || std::string(argv[1]) == "--help") {
            std::cerr << "usage: vc_object_convert SOURCE.vcobj DESTINATION.vcobj [MAPPING.json [REPORT.json]]\n"
                         "       vc_object_convert --in-place STORE.vcobj [MAPPING.json [REPORT.json]]\n"
                         "       vc_object_convert --enable-history STORE.vcobj\n"
                         "       vc_object_convert --remove-history-and-disable STORE.vcobj\n"
                         "Conversion copies an upstream schema 16–20 SOURCE (never modified) into a new DESTINATION.\n"
                         "Stop every client first; settle legacy hub replicas with their old tooling.\n"
                         "MAPPING defaults to DESTINATION.mapping.json, REPORT to DESTINATION.report.json.\n"
                         "History maintenance validates a sibling file before atomically replacing STORE.\n";
            return 2;
        }
        const bool inPlace = std::string(argv[1]) == "--in-place";
        if (!inPlace && std::string(argv[1]).starts_with("--"))
            throw std::invalid_argument("unknown converter option: " + std::string(argv[1]));
        const std::string destination = argv[2];
        const std::string mappingPath = argc > 3 ? argv[3] : destination + ".mapping.json";
        const std::string reportPath = argc > 4 ? argv[4] : destination + ".report.json";
        const auto destinationKey = std::filesystem::weakly_canonical(destination);
        const auto mappingKey = std::filesystem::weakly_canonical(mappingPath);
        const auto reportKey = std::filesystem::weakly_canonical(reportPath);
        if (destinationKey == mappingKey || destinationKey == reportKey || mappingKey == reportKey)
            throw std::runtime_error("destination, mapping and report must be distinct files");
        for (const auto& path : {mappingPath, reportPath})
            if (std::filesystem::exists(path) || std::filesystem::is_symlink(path))
                throw std::runtime_error("output already exists: " + path);
        const auto reports = [&](const auto& result) {
            vc::objects::migration::writeConversionReports(result, mappingPath, reportPath);
        };
        const auto result = inPlace ? vc::objects::migration::convertStoreInPlace(destination, reports)
                                    : vc::objects::migration::convertStore(argv[1], destination);
        if (!inPlace) reports(result);
        std::cout << result.reportJson << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "vc_object_convert: " << error.what() << '\n';
        return 1;
    }
}
