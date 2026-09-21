// Optional local-data regression/benchmark tool. No hard-coded dataset paths.
#include "vc/lasagna/RobustLineStrip.hpp"
#include "vc/lasagna/LineViewBuilder.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <chrono>
#include <iostream>

int main(int argc,char** argv) {
    try {
        if(argc!=3&&argc!=6&&argc!=7)throw std::runtime_error("Usage: robust_line_strip_probe fiber.json output-prefix [seed.bin rows cols [preserve-width]]");
        const auto started=std::chrono::steady_clock::now();std::ifstream in(argv[1]);auto j=nlohmann::json::parse(in);
        vc::lasagna::LasagnaDatasetOpenOptions options;options.workingToBaseScale=j.at("normal_scale");
        auto ds=vc::lasagna::LasagnaDataset::openLocation(j.at("normal_manifest").get<std::string>(),options);vc::lasagna::LasagnaNormalSampler sampler(ds);
        cv::Mat_<cv::Vec3f> grid;std::vector<double> arc;
        if(argc>=6) {
            grid.create(std::stoi(argv[4]),std::stoi(argv[5]));std::ifstream seed(argv[3],std::ios::binary);
            seed.read(reinterpret_cast<char*>(grid.data),std::streamsize(grid.total()*sizeof(cv::Vec3f)));
            if(!seed)throw std::runtime_error("Cannot read complete seed grid");
            grid=vc::lasagna::fitRobustLineStrip(grid,sampler,argc==7&&std::string(argv[6])=="preserve-width");
        } else {
            vc::lasagna::LineModel line;
            for(const auto& v:j.at("line")){cv::Vec3d p{v[0],v[1],v[2]};line.points.push_back({p,sampler.sampleNormal(p),true});}
            vc::lasagna::LineViewConfig config;config.crossSpacingBaseVoxels=2;config.crossSampleCount=61;config.targetSpacingBaseVoxels=2;
            config.controlPointLinePositions={0.,double(line.points.size()-1)};config.normalSampler=&sampler;config.buildLineZSlices=false;
            config.crossSpacingBaseVoxels=j.value("output_spacing",2.);config.targetSpacingBaseVoxels=config.crossSpacingBaseVoxels;
            config.crossSampleCount=j.value("output_rows",61);
            const auto result=vc::lasagna::buildLineViewSurfaces(line,config);grid=result.lineSurface->rawPointsPtr()->clone();arc=result.stripPositionMap.stripGridArclengths;
        }
        std::ofstream binary(std::string(argv[2])+".bin",std::ios::binary);binary.write(reinterpret_cast<const char*>(grid.data),std::streamsize(grid.total()*sizeof(cv::Vec3f)));
        const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
        const nlohmann::json result={{"shape",{grid.rows,grid.cols,3}},{"arc",arc},{"seconds",seconds},{"strip_generation",vc::lasagna::kRobustLineStripVersion}};
        std::ofstream(std::string(argv[2])+".json")<<result.dump();std::cout<<seconds<<" seconds\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
