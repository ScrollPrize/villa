// Tests the actual Zarr reader and bounded previews, including missing pyramids.
#include "VolumePreview.hpp"

static void testVolumePreview(const std::filesystem::path& root)
{
    using namespace vc3d::scene3d;
    auto require=[](bool ok, const char* message) { if (!ok) throw std::runtime_error(message); };
    vc::test::ZarrVolumeSpec spec;
    spec.shapeZYX={24,24,24}; spec.chunkShapeZYX={8,8,8}; spec.numLevels=3;
    spec.dtype=vc::render::ChunkDtype::UInt16;
    auto volume=vc::test::createZarrVolume(root/"preview.zarr",spec);
    Array3D<uint16_t> coarse({6,6,6}, uint16_t(1024));
    vc::test::writeZYX(*volume,coarse,{0,0,0},2);
    std::atomic_bool stop{false};
    const size_t budget=8*1024*1024;
    auto data=sampleVolume(*volume,2,budget,stop);
    require(data && data->images.front()->level==2 && !data->resampled, "preview honors a source level that fits");
    const auto& pixels=data->images.front()->slab->pixels;
    for (size_t i=0; i<pixels.size(); i+=2)
        require(pixels[i]==4 && pixels[i+1]==255,"preview shares CT uint16 normalization, including pyramid edges");
    auto fallback=sampleVolume(*volume,0,4096,stop);
    require(fallback->requestedLevel==0 && fallback->images.front()->level==2,"preview chooses an available coarser level to fit");
    auto reduced=sampleVolume(*volume,0,512,stop);
    require(reduced && reduced->resampled,"preview resamples when even the coarsest level is too large");
    require(2*reduced->images.front()->slab->pixels.size()<=512,"CPU and GPU textures fit a small budget");
    bool rejected=false;
    try { sampleVolume(*volume,0,1,stop); } catch (const std::invalid_argument&) { rejected=true; }
    require(rejected,"invalid byte budgets are rejected");
    stop=true; rejected=false;
    try { sampleVolume(*volume,0,budget,stop); } catch (const std::runtime_error&) { rejected=true; }
    require(rejected,"preview sampling can be cancelled");
    stop=false;
    coarse.fill(0);
    vc::test::writeZYX(*volume,coarse,{0,0,0},2);
    require(!sampleVolume(*volume,2,budget,stop),"empty CT produces no visible preview");

    // A noisy, single-level volume previously generated excessive contour
    // geometry. It must now render with no pyramid and a tiny texture budget.
    spec.shapeZYX={48,32,64}; spec.numLevels=1; spec.dtype=vc::render::ChunkDtype::UInt8;
    auto single=vc::test::createZarrVolume(root/"no-pyramid.zarr",spec);
    Array3D<uint8_t> values({48,32,64});
    for (int z=0; z<48; ++z)
        for (int y=0; y<32; ++y)
            for (int x=0; x<64; ++x) values(z,y,x)=(x+y+z)%2 ? 200 : 40;
    vc::test::writeZYX(*single,values,{0,0,0});
    data=sampleVolume(*single,3,4096,stop);
    require(data && data->images.front()->level==0 && data->resampled,"single-level noisy CT works without contour geometry");
    const auto& slab=*data->images.front()->slab;
    require(2*slab.pixels.size()<=4096,"small-budget preview has bounded CPU and GPU storage");
    require(slab.width<=256 && slab.height<=256 && slab.depth<=256,"preview respects portable GPU texture limits");
    require(slab.origin==Point{-.5,-.5,-.5} && slab.right==Point{64,0,0} && slab.down==Point{0,32,0} && slab.through==Point{0,0,48},
            "downsampling preserves physical size and voxel registration on every axis");
    require(data->bounds.lower==Point{0,0,0} && data->bounds.upper==Point{63,31,47},"camera bounds use native CT coordinates");
    const auto repeat=sampleVolume(*single,3,4096,stop);
    require(repeat->images.front()->slab->pixels==slab.pixels,"preview sampling is deterministic");

    // More than 256 source voxels on one axis: the entire length remains
    // visible even with no coarser Zarr level to fall back to.
    spec.shapeZYX={300,4,8}; spec.chunkShapeZYX={32,4,8};
    auto tall=vc::test::createZarrVolume(root/"tall.zarr",spec);
    Array3D<uint8_t> solid({300,4,8},uint8_t(150));
    vc::test::writeZYX(*tall,solid,{0,0,0});
    data=sampleVolume(*tall,0,budget,stop);
    require(data && data->resampled && data->images.front()->slab->depth==256,"long single-level volumes are reduced automatically");
    require(data->images.front()->slab->through==Point{0,0,300},"texture edge limit does not shrink the volume");
    auto detailed=sampleVolume(*tall,0,budget,stop,512);
    require(detailed && !detailed->resampled && detailed->images.front()->slab->depth==300,
            "raising resolution retains source detail above the former 256-voxel limit");
    require(detailed->bounds.upper==data->bounds.upper && detailed->images.front()->slab->through==data->images.front()->slab->through,
            "increasing resolution preserves native geometry");
    auto limited=sampleVolume(*tall,0,4096,stop,1024);
    require(limited && limited->resampled && 2*limited->images.front()->slab->pixels.size()<=4096-4096/4,
            "higher requested resolution still respects texture and scratch memory budget");
    rejected=false;
    try { sampleVolume(*tall,0,budget,stop,0); } catch (const std::invalid_argument&) { rejected=true; }
    require(rejected,"invalid resolutions are rejected");
}
