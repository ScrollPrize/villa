// Reproduce the HTJ2K fixtures with OpenJPH 0.32.0; no HDF5 filter plugin required.
// g++ -O2 generate.cpp -I<OpenJPH>/src/core/openjph <build>/src/core/libopenjph.a -o generate
// ./generate <output directory>
#include <ojph_codestream.h>
#include <ojph_file.h>
#include <ojph_mem.h>
#include <ojph_params.h>
#include <cstdint>
#include <string>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    for (bool signed32 : {true, false}) for (bool reversible : {true, false}) {
        const int width = signed32 ? 259 : 256, height = signed32 ? 257 : 256;
        ojph::j2c_outfile file;
        const auto path = std::string(argv[1]) + (signed32 ? "/signed32_" : "/unsigned16_") +
                          (reversible ? "53.j2k" : "97.j2k");
        file.open(path.c_str());
        ojph::codestream cs;
        auto siz = cs.access_siz();
        siz.set_image_extent(ojph::point(width, height));
        siz.set_num_components(1);
        siz.set_component(0, ojph::point(1, 1), signed32 ? 32 : 16, signed32);
        auto cod = cs.access_cod();
        cod.set_num_decomposition(3);
        cod.set_block_dims(64, 64);
        cod.set_progression_order("RPCL");
        cod.set_color_transform(false);
        cod.set_reversible(reversible);
        cs.set_planar(true);
        cs.write_headers(&file);
        ojph::ui32 comp;
        auto* line = cs.exchange(nullptr, comp);
        std::uint32_t state = 17;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                state = 1664525u * state + 1013904223u;
                line->i32[x] = signed32 ? static_cast<std::int32_t>(state >> 6) - (1 << 25) : 20000 + (state >> 17);
            }
            line = cs.exchange(line, comp);
        }
        cs.flush();
        cs.close();
    }
}
