// Low-level test for blosc2 + H264 decompression
// Usage: test_blosc2_h264 <chunk_file>

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>

#include <blosc2.h>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <chunk_file>\n";
        std::cerr << "Example: " << argv[0] << " /path/to/zarr/0/100/50/70\n";
        return 1;
    }

    const char* chunk_path = argv[1];
    std::cout << "Chunk file: " << chunk_path << "\n\n";

    // Initialize blosc2
    std::cout << "Initializing blosc2...\n";
    blosc2_init();

    // List registered codecs
    std::cout << "Checking for openh264 codec...\n";
    int openh264_code = blosc2_compname_to_compcode("openh264");
    std::cout << "  openh264 codec code: " << openh264_code << "\n";
    if (openh264_code < 0) {
        std::cerr << "  ERROR: openh264 codec not found!\n";
        blosc2_destroy();
        return 1;
    }

    // Read chunk file
    std::cout << "Reading chunk file...\n";
    std::ifstream file(chunk_path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "  ERROR: Cannot open file\n";
        blosc2_destroy();
        return 1;
    }

    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> compressed(file_size);
    if (!file.read(reinterpret_cast<char*>(compressed.data()), file_size)) {
        std::cerr << "  ERROR: Cannot read file\n";
        blosc2_destroy();
        return 1;
    }
    std::cout << "  File size: " << file_size << " bytes\n";

    // Parse blosc2 header
    std::cout << "\nParsing blosc2 header...\n";
    if (file_size < 32) {
        std::cerr << "  ERROR: File too small for blosc2 header\n";
        blosc2_destroy();
        return 1;
    }

    // Get info about the compressed chunk
    int32_t nbytes, cbytes, blocksize;
    blosc2_cbuffer_sizes(compressed.data(), &nbytes, &cbytes, &blocksize);
    std::cout << "  Uncompressed size (nbytes): " << nbytes << "\n";
    std::cout << "  Compressed size (cbytes): " << cbytes << "\n";
    std::cout << "  Block size: " << blocksize << "\n";

    // Get more info
    const char* compname = blosc2_cbuffer_complib(compressed.data());
    std::cout << "  Compressor name: " << (compname ? compname : "NULL") << "\n";

    // Try to decompress
    std::cout << "\nAttempting decompression...\n";
    std::vector<uint8_t> decompressed(nbytes);

    int result = blosc2_decompress(compressed.data(), (int32_t)file_size,
                                   decompressed.data(), nbytes);

    if (result < 0) {
        std::cerr << "  ERROR: Decompression failed with code " << result << "\n";

        // Try with explicit context
        std::cout << "\nTrying with explicit decompression context...\n";
        blosc2_dparams dparams = BLOSC2_DPARAMS_DEFAULTS;
        dparams.nthreads = 1;
        blosc2_context* dctx = blosc2_create_dctx(dparams);
        if (dctx == NULL) {
            std::cerr << "  ERROR: Failed to create decompression context\n";
        } else {
            result = blosc2_decompress_ctx(dctx, compressed.data(), (int32_t)file_size,
                                           decompressed.data(), nbytes);
            if (result < 0) {
                std::cerr << "  ERROR: Context decompression failed with code " << result << "\n";
            }
            blosc2_free_ctx(dctx);
        }
    }

    if (result >= 0) {
        std::cout << "  Decompression successful! " << result << " bytes\n";

        // Analyze decompressed data
        size_t zero_count = 0;
        size_t nonzero_count = 0;
        uint8_t min_val = 255, max_val = 0;
        double sum = 0;

        for (size_t i = 0; i < (size_t)result; i++) {
            uint8_t val = decompressed[i];
            if (val == 0) zero_count++;
            else nonzero_count++;
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
            sum += val;
        }

        std::cout << "\n  Decompressed data statistics:\n";
        std::cout << "    Total bytes: " << result << "\n";
        std::cout << "    Zero bytes: " << zero_count << " (" << 100.0 * zero_count / result << "%)\n";
        std::cout << "    Non-zero bytes: " << nonzero_count << " (" << 100.0 * nonzero_count / result << "%)\n";
        std::cout << "    Min value: " << (int)min_val << "\n";
        std::cout << "    Max value: " << (int)max_val << "\n";
        std::cout << "    Mean value: " << sum / result << "\n";

        // Print first 64 bytes
        std::cout << "\n  First 64 decompressed bytes:\n    ";
        for (int i = 0; i < 64 && i < result; i++) {
            printf("%02x ", decompressed[i]);
            if ((i + 1) % 16 == 0) std::cout << "\n    ";
        }
        std::cout << "\n";
    }

    // Print raw header bytes for debugging
    std::cout << "\nRaw file header (first 128 bytes):\n";
    for (int i = 0; i < 128 && i < (int)file_size; i++) {
        printf("%02x ", compressed[i]);
        if ((i + 1) % 16 == 0) std::cout << "\n";
    }
    std::cout << "\n";

    // Decode the blosc2 header manually
    std::cout << "\nManual header decode:\n";
    uint8_t version = compressed[0];
    uint8_t versionlz = compressed[1];
    uint8_t flags = compressed[2];
    uint8_t typesize_byte = compressed[3];

    // For blosc2, compcode is encoded in bits 5-7 of flags for codes 0-7
    // For user-defined codecs (>7), it uses an extended mechanism
    uint8_t compcode_low = (flags >> 5) & 0x7;

    std::cout << "  version: " << (int)version << "\n";
    std::cout << "  versionlz: " << (int)versionlz << "\n";
    std::cout << "  flags byte: 0x" << std::hex << (int)flags << std::dec << "\n";
    std::cout << "  typesize byte: " << (int)typesize_byte << "\n";
    std::cout << "  compcode (bits 5-7): " << (int)compcode_low << "\n";

    // Check if this is a blosc2 frame/superchunk
    // Byte 4-7 should be nbytes
    uint32_t nbytes_header = *(uint32_t*)(compressed.data() + 4);
    std::cout << "  nbytes from header bytes 4-7: " << nbytes_header << "\n";

    // For extended codec IDs, check if there's a special marker
    // In blosc2, when compcode_low == 6 or 7, it might indicate a user codec
    // Let's check if there's an extended codec byte
    if (compcode_low >= 6) {
        std::cout << "  Extended codec ID detected (compcode_low >= 6)\n";
        // The actual codec ID might be stored elsewhere
        // Let's check byte 3 (typesize) - sometimes used for extended info
        std::cout << "  Possible extended codec info in byte 3: " << (int)typesize_byte << "\n";
    }

    blosc2_destroy();
    return (result >= 0) ? 0 : 1;
}
