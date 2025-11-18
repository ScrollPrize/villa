#include "vc/ui/VCCollection.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/Slicing.hpp"
#include <opencv2/imgcodecs.hpp>
#include <boost/program_options.hpp>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>

namespace po = boost::program_options;

static inline cv::Vec2f mul(const cv::Vec2f &a, const cv::Vec2f &b)
{
    return{a[0]*b[0],a[1]*b[1]};
}

template <typename E>
float ldist(const E &p, const cv::Vec3f &tgt_o, const cv::Vec3f &tgt_v)
{
    return cv::norm((p-tgt_o).cross(p-tgt_o-tgt_v))/cv::norm(tgt_v);
}

template <typename E>
static float search_min_line(const cv::Mat_<E> &points, cv::Vec2f &loc, cv::Vec3f &out, cv::Vec3f tgt_o, cv::Vec3f tgt_v, cv::Vec2f init_step, float min_step_x)
{
    cv::Rect boundary(1,1,points.cols-2,points.rows-2);
    if (!boundary.contains(cv::Point(loc))) {
        out = {-1,-1,-1};
        return -1;
    }
    
    bool changed = true;
    E val = at_int(points, loc);
    out = val;
    float best = ldist(val, tgt_o, tgt_v);
    float res;
    
    std::vector<cv::Vec2f> search = {{0,-1},{0,1},{-1,-1},{-1,0},{-1,1},{1,-1},{1,0},{1,1}};
    cv::Vec2f step = init_step;
    
    while (changed) {
        changed = false;
        
        for(auto &off : search) {
            cv::Vec2f cand = loc+mul(off,step);
            
            if (!boundary.contains(cv::Point(cand)))
                continue;
                
                val = at_int(points, cand);
                res = ldist(val, tgt_o, tgt_v);
                if (res < best) {
                    changed = true;
                    best = res;
                    loc = cand;
                    out = val;
                }
        }
        
        if (changed)
            continue;
        
        step *= 0.5;
        changed = true;
        
        if (step[0] < min_step_x)
            break;
    }
    
    return best;
}

template <typename E>
float line_off(const E &p, const cv::Vec3f &tgt_o, const cv::Vec3f &tgt_v)
{
    return (tgt_o-p).dot(tgt_v)/cv::norm(tgt_v);
}

using IntersectVec = std::vector<std::pair<float,cv::Vec2f>>;

IntersectVec getIntersects(const cv::Vec2i &seed, const cv::Mat_<cv::Vec3f> &points, const cv::Vec2f &step)
{
    cv::Vec3f o = points(seed[1],seed[0]);
    cv::Vec3f n = grid_normal(points, {(float)seed[0],(float)seed[1],0});
    if (std::isnan(n[0]))
        return {};
    std::vector<cv::Vec2f> locs = {seed};
    uint32_t sr = seed[1];
    for(int i=0;i<1000;i++)
    {
        cv::Vec2f loc = {static_cast<float>(rand_r(&sr) % points.cols), static_cast<float>(seed[1] - 50 + (rand_r(&sr) % 100))};
        cv::Vec3f res;
        float dist = search_min_line(points, loc, res, o, n, step, 0.01);

        if (dist > 0.5 || dist < 0)
            continue;

        if (!loc_valid_xy(points,loc))
            continue;

        // std::cout << dist << res << loc << std::endl;

        bool found = false;
        for(auto l : locs) {
            if (cv::norm(loc, l) <= 4) {
                found = true;
                break;
            }
        }
        if (!found)
            locs.push_back(loc);
    }

    IntersectVec dist_locs;
    for(auto l : locs)
        dist_locs.push_back({line_off(at_int(points,l),o,n), l});
    
    std::sort(dist_locs.begin(), dist_locs.end(), [](auto a, auto b) {return a.first < b.first; });
    return dist_locs;    
}


int main(int argc, char** argv) {
    po::options_description desc("Generate relative winding annotations from a .tiffxyz file.");
    desc.add_options()
        ("help,h", "Print help")
        ("input", po::value<std::string>(), "Input surface file (.tiffxyz)")
        ("winding", po::value<std::string>(), "Input winding file (.tif)")
        ("output", po::value<std::string>(), "Output VCCollection file (.json)")
        ("num-collections", po::value<int>()->default_value(10), "Number of random collections to generate");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    if (!vm.count("input") || !vm.count("winding") || !vm.count("output")) {
        std::cerr << "Error: --input, --winding, and --output are required." << std::endl;
        return 1;
    }

    std::string input_path = vm["input"].as<std::string>();
    std::string winding_path = vm["winding"].as<std::string>();
    std::string output_path = vm["output"].as<std::string>();
    int num_collections = vm["num-collections"].as<int>();

    QuadSurface* surface = load_quad_from_tifxyz(input_path);
    if (!surface) {
        std::cerr << "Error: Failed to load surface from " << input_path << std::endl;
        return 1;
    }

    cv::Mat_<float> winding = cv::imread(winding_path, cv::IMREAD_UNCHANGED);
    if (winding.empty()) {
        std::cerr << "Error: Failed to load winding from " << winding_path << std::endl;
        return 1;
    }

    cv::Mat_<cv::Vec3f> points = surface->rawPoints();
    
    VCCollection collection;

    std::cout << "wtf "  << std::endl;
    for (int i = 0; i < num_collections; ++i) {
        cv::Point seed_loc(rand() % points.cols, rand() % points.rows);

        std::cout << "try " << seed_loc << std::endl;

        if (points(seed_loc.y, seed_loc.x)[0] == -1) {
            i--; // Try again with a new random point
            continue;
        }

        IntersectVec intersects = getIntersects({seed_loc.x, seed_loc.y}, points, surface->scale());

        std::cout << "got " << intersects.size() << std::endl;

        if (intersects.empty()) {
            continue;
        }

        std::string collection_name = collection.generateNewCollectionName("col");
        collection.addCollection(collection_name);

        float base_winding = at_int(winding, {float(seed_loc.x), float(seed_loc.y)});
        float last_winding = -1e9;
        
        std::vector<std::pair<float, cv::Vec2f>> filtered_intersects;
        for (const auto& intersect : intersects) {
            cv::Vec2f loc = intersect.second;
            float w = at_int(winding, loc);
            
            float diff = std::abs(w - base_winding);
            if (std::abs(diff - round(diff)) < 0.1) {
                filtered_intersects.push_back({w, loc});
            }
        }

        std::sort(filtered_intersects.begin(), filtered_intersects.end(), [](auto a, auto b) {return a.first < b.first; });

        for (const auto& intersect : filtered_intersects) {
            float w = intersect.first;
            cv::Vec2f loc = intersect.second;

            int current_winding_int = round(w);
            if (current_winding_int != round(last_winding)) {
                 ColPoint pt = collection.addPoint(collection_name, at_int(points, loc));
                 pt.winding_annotation = w;
                 collection.updatePoint(pt);
                 last_winding = w;
            }
        }
    }
    
    collection.saveToJSON(output_path);

    delete surface;

    std::cout << "Successfully generated annotations and saved to " << output_path << std::endl;

    return 0;
}
