#include "vc_object_store/objects/StoreConversion.hpp"
#include <iostream>
int main(int argc, char** argv) try
{
    if (argc==3) {
        const std::string option=argv[1];
        if (option!="--enable-history" && option!="--remove-history-and-disable") throw std::invalid_argument("unknown maintenance option");
        vc::objects::changeHistoryMode(argv[2],option=="--enable-history" ? vc::objects::HistoryMode::On : vc::objects::HistoryMode::Off);
        std::cout<<"History policy updated atomically\n"; return 0;
    }
    if (argc != 2) { std::cerr << "usage: vc_object_convert [--enable-history|--remove-history-and-disable] STORE.vcobj\nOffline maintenance; replaces STORE after validation. Without an option, converts schema 17 to v2.\n"; return 2; }
    const auto result = vc::objects::convertV1ToV2(argv[1]);
    std::cout << "Verified " << result.tables << " tables, " << result.rows << " rows; v2 size " << result.bytes << " bytes\n";
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
