// Offline conversion of a schema-17 or schema-18 object store to the current schema.
#include "vc_object_store/migration/ConvertStore.hpp"

#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char** argv)
{
    if (argc < 3 || argc > 5 || std::string(argv[1]) == "--help") {
        std::cerr << "usage: vc_object_convert SOURCE.vcobj DESTINATION.vcobj [MAPPING.json [REPORT.json]]\n"
                     "Copies SOURCE (never modified) into the new DESTINATION with numbered history.\n"
                     "Stop every client of SOURCE first; settle hub replicas with the old tooling.\n"
                     "MAPPING defaults to DESTINATION.mapping.json, REPORT to DESTINATION.report.json.\n";
        return 2;
    }
    const std::string destination = argv[2];
    const std::string mappingPath = argc > 3 ? argv[3] : destination + ".mapping.json";
    const std::string reportPath = argc > 4 ? argv[4] : destination + ".report.json";
    for (const auto& path : {mappingPath, reportPath})
        if (std::filesystem::exists(path)) { std::cerr << "vc_object_convert: " << path << " exists\n"; return 1; }
    try {
        const auto result = vc::objects::migration::convertStore(argv[1], destination);
        std::ofstream(mappingPath) << result.mappingJson << "\n";
        std::ofstream(reportPath) << result.reportJson << "\n";
        std::cout << result.reportJson << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "vc_object_convert: " << error.what() << "\n";
        return 1;
    }
}
