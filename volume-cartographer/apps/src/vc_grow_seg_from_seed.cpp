#include <random>

#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/types/ChunkedTensor.hpp"
#include "vc/core/util/StreamOperators.hpp"
#include "vc/tracer/Tracer.hpp"


#include "z5/factory.hxx"
#include <nlohmann/json.hpp>
#include <boost/program_options.hpp>

#include <omp.h>

#include <algorithm>
#include <cmath>
 
namespace po = boost::program_options;
using shape = z5::types::ShapeType;


using json = nlohmann::json;



std::string time_str()
{
    using namespace std::chrono;
    auto now = system_clock::now();
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
    auto timer = system_clock::to_time_t(now);
    std::tm bt = *std::localtime(&timer);
    
    std::ostringstream oss;
    oss << std::put_time(&bt, "%Y%m%d%H%M%S");
    oss << std::setfill('0') << std::setw(3) << ms.count();

    return oss.str();
}

template <typename T, typename I>
float get_val(I &interp, cv::Vec3d l) {
    T v;
    interp.Evaluate(l[2], l[1], l[0], &v);
    return v;
}

bool check_existing_segments(const std::filesystem::path& tgt_dir, const cv::Vec3d& origin,
                           const std::string& name_prefix, int search_effort) {
    for (const auto& entry : std::filesystem::directory_iterator(tgt_dir)) {
        if (!std::filesystem::is_directory(entry)) {
            continue;
        }

        std::string name = entry.path().filename();
        if (name.compare(0, name_prefix.size(), name_prefix)) {
            continue;
        }

        std::filesystem::path meta_fn = entry.path() / "meta.json";
        if (!std::filesystem::exists(meta_fn)) {
            continue;
        }

        std::ifstream meta_f(meta_fn);
        json meta = json::parse(meta_f);

        if (!meta.count("bbox") || meta.value("format","NONE") != "tifxyz") {
            continue;
        }

        SurfaceMeta other(entry.path(), meta);
        if (contains(other, origin, search_effort)) {
            std::cout << "Found overlapping segment at location: " << entry.path() << std::endl;
            return true;
        }
    }
    return false;
}

static auto load_direction_fields(json const&params, ChunkCache *chunk_cache, std::filesystem::path const &cache_root)
{
    std::vector<DirectionField> direction_fields;
    if (params.contains("direction_fields")) {
        if (!params["direction_fields"].is_array()) {
            std::cerr << "WARNING: direction_fields must be an array; ignoring" << std::endl;
        }
        for (auto const& direction_field : params["direction_fields"]) {
            std::string const zarr_path = direction_field["zarr"];
            std::string const direction = direction_field["dir"];
            if (!std::ranges::contains(std::vector{"horizontal", "vertical", "normal"}, direction)) {
                std::cerr << "WARNING: invalid direction in direction_field " << zarr_path << "; skipping" << std::endl;
                continue;
            }
            int const ome_scale = direction_field["scale"];
            float scale_factor = std::pow(2, -ome_scale);
            z5::filesystem::handle::Group dirs_group(zarr_path, z5::FileMode::FileMode::r);
            std::vector<std::unique_ptr<z5::Dataset>> direction_dss;
            for (auto dim : std::string("xyz")) {
                z5::filesystem::handle::Group dim_group(dirs_group, std::string(&dim, 1));
                z5::filesystem::handle::Dataset dirs_ds_handle(dim_group, std::to_string(ome_scale), ".");
                direction_dss.push_back(z5::filesystem::openDataset(dirs_ds_handle));
            }
            std::cout << "direction field dataset shape " << direction_dss.front()->shape() << std::endl;
            std::unique_ptr<z5::Dataset> maybe_weight_ds;
            if (direction_field.contains("weight_zarr")) {
                std::string const weight_zarr_path = direction_field["weight_zarr"];
                z5::filesystem::handle::Group weight_group(weight_zarr_path);
                z5::filesystem::handle::Dataset weight_ds_handle(weight_group, std::to_string(ome_scale), ".");
                maybe_weight_ds = z5::filesystem::openDataset(weight_ds_handle);
            }
            std::string const unique_id = std::to_string(std::hash<std::string>{}(dirs_group.path().string() + std::to_string(ome_scale)));
            float weight = 1.0f;
            if (direction_field.contains("weight")) {
                try {
                    weight = std::clamp(direction_field["weight"].get<float>(), 0.0f, 10.0f);
                } catch (const std::exception& ex) {
                    std::cerr << "WARNING: invalid weight for direction field " << zarr_path << ": " << ex.what() << std::endl;
                }
            }

            direction_fields.emplace_back(
                direction,
                std::make_unique<Chunked3dVec3fFromUint8>(std::move(direction_dss), scale_factor, chunk_cache, cache_root, unique_id),
                maybe_weight_ds ? std::make_unique<Chunked3dFloatFromUint8>(std::move(maybe_weight_ds), scale_factor, chunk_cache, cache_root, unique_id + "_conf") : std::unique_ptr<Chunked3dFloatFromUint8>(),
                weight);
        }
    }
    return direction_fields;
}

int main(int argc, char *argv[])
{
    std::filesystem::path vol_path, tgt_dir, params_path, resume_path, correct_path;
    cv::Vec3d origin;
    json params;
    VCCollection corrections;
    bool skip_overlap_check = false;

    bool use_old_args = (argc == 4 || argc == 7) && argv[1][0] != '-' && argv[2][0] != '-' && argv[3][0] != '-';

    if (use_old_args) {
        vol_path = argv[1];
        tgt_dir = argv[2];
        params_path = argv[3];
        if (argc == 7) {
            origin = {atof(argv[4]), atof(argv[5]), atof(argv[6])};
        }
    } else {
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help,h", "produce help message")
            ("volume,v", po::value<std::string>()->required(), "OME-Zarr volume path")
            ("target-dir,t", po::value<std::string>()->required(), "Target directory for output")
            ("params,p", po::value<std::string>()->required(), "JSON parameters file")
            ("seed,s", po::value<std::vector<float>>()->multitoken(), "Seed coordinates (x y z)")
            ("resume", po::value<std::string>(), "Path to a tifxyz surface to resume from")
            ("rewind-gen", po::value<int>(), "Generation to rewind to")
            ("correct", po::value<std::string>(), "JSON file with point-based corrections for resume mode")
            ("skip-overlap-check", "Do not perform overlap check with other surfaces after tracing")
            ("inpaint", "perform automatic inpainting on all detected holes.");

        po::variables_map vm;
        try {
            po::store(po::parse_command_line(argc, argv, desc), vm);

            if (vm.count("help")) {
                std::cout << desc << std::endl;
                return EXIT_SUCCESS;
            }

            po::notify(vm);
        } catch (const po::error &e) {
            std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
            std::cerr << desc << std::endl;
            return EXIT_FAILURE;
        }

        vol_path = vm["volume"].as<std::string>();
        tgt_dir = vm["target-dir"].as<std::string>();
        params_path = vm["params"].as<std::string>();

        if (vm.count("seed")) {
            auto seed_coords = vm["seed"].as<std::vector<float>>();
            if (seed_coords.size() != 3) {
                std::cerr << "ERROR: --seed requires exactly 3 coordinates (x y z)" << std::endl;
                return EXIT_FAILURE;
            }
            origin = {seed_coords[0], seed_coords[1], seed_coords[2]};
        }
        if (vm.count("resume")) {
            resume_path = vm["resume"].as<std::string>();
        }

        if (vm.count("correct")) {
            if (!vm.count("resume")) {
                std::cerr << "ERROR: --correct can only be used with --resume" << std::endl;
                return EXIT_FAILURE;
            }
            correct_path = vm["correct"].as<std::string>();
            std::ifstream correct_f(correct_path.string());
            if (!corrections.loadFromJSON(correct_path.string())) {
                std::cerr << "ERROR: Could not load or parse corrections file: " << correct_path << std::endl;
                return EXIT_FAILURE;
            }
        }
        
        std::ifstream params_f(params_path.string());
        params = json::parse(params_f);

        if (vm.count("rewind-gen")) {
            params["rewind_gen"] = vm["rewind-gen"].as<int>();
        }

        if (vm.count("inpaint")) {
            params["inpaint"] = true;
        }

        if (vm.count("skip-overlap-check")) {
            skip_overlap_check = true;
        }
    }

    if (params.empty()) {
        std::ifstream params_f(params_path.string());
        params = json::parse(params_f);
    }

    // Honor optional CUDA toggle from params (default true)
    if (params.contains("use_cuda")) {
        set_space_tracing_use_cuda(params.value("use_cuda", true));
    } else {
        set_space_tracing_use_cuda(true);
    }

    z5::filesystem::handle::Group group(vol_path, z5::FileMode::FileMode::r);
    z5::filesystem::handle::Dataset ds_handle(group, "0", json::parse(std::ifstream(vol_path/"0/.zarray")).value<std::string>("dimension_separator","."));
    std::unique_ptr<z5::Dataset> ds = z5::filesystem::openDataset(ds_handle);

    std::cout << "zarr dataset size for scale group 0 " << ds->shape() << std::endl;
    std::cout << "chunk shape shape " << ds->chunking().blockShape() << std::endl;

    ChunkCache chunk_cache(params.value("cache_size", 1e9));

    passTroughComputor pass;
    Chunked3d<uint8_t,passTroughComputor> tensor(pass, ds.get(), &chunk_cache);
    CachedChunked3dInterpolator<uint8_t,passTroughComputor> interpolator(tensor);

    auto chunk_size = ds->chunking().blockShape();

    srand(clock());

    std::string name_prefix = "auto_grown_";
    int tgt_overlap_count = params.value("tgt_overlap_count", 20);
    float min_area_cm = params.value("min_area_cm", 0.3);
    int search_effort = params.value("search_effort", 10);
    int thread_limit = params.value("thread_limit", 0);

    float voxelsize = json::parse(std::ifstream(vol_path/"meta.json"))["voxelsize"];
    
    std::filesystem::path cache_root;
    if (params.contains("cache_root") && params["cache_root"].is_string()) {
        cache_root = params["cache_root"].get<std::string>();
    } else if (params.contains("cache_root") && !params["cache_root"].is_null()) {
        std::cerr << "WARNING: cache_root must be a string; ignoring" << std::endl;
    }

    std::string mode = params.value("mode", "seed");
    
    std::cout << "mode: " << mode << std::endl;
    std::cout << "step size: " << params.value("step_size", 20.0f) << std::endl;
    std::cout << "min_area_cm: " << min_area_cm << std::endl;
    std::cout << "tgt_overlap_count: " << tgt_overlap_count << std::endl;

    auto direction_fields = load_direction_fields(params, &chunk_cache, cache_root);

    std::unordered_map<std::string,SurfaceMeta*> surfs;
    std::vector<SurfaceMeta*> surfs_v;
    SurfaceMeta *src;

    //expansion mode
    int count_overlap = 0;
    if (mode == "expansion") {
        //got trough all exising segments (that match filter/start with auto ...)
        //list which ones do not yet less N overlapping (in symlink dir)
        //shuffle
        //iterate and for every one
            //select a random point (close to edge?)
            //check against list if other surf in bbox if we can find the point
            //if yes add symlinkg between the two segs
            //if both still have less than N then grow a seg from the seed
            //after growing, check locations on the new seg agains all existing segs

        for (const auto& entry : std::filesystem::directory_iterator(tgt_dir))
            if (std::filesystem::is_directory(entry)) {
                std::string name = entry.path().filename();
                if (name.compare(0, name_prefix.size(), name_prefix))
                    continue;

                std::cout << entry.path() << entry.path().filename() << std::endl;

                std::filesystem::path meta_fn = entry.path() / "meta.json";
                if (!std::filesystem::exists(meta_fn))
                    continue;

                std::ifstream meta_f(meta_fn);
                json meta = json::parse(meta_f);

                if (!meta.count("bbox"))
                    continue;

                if (meta.value("format","NONE") != "tifxyz")
                    continue;

                SurfaceMeta *sm = new SurfaceMeta(entry.path(), meta);
                sm->readOverlapping();

                surfs[name] = sm;
                surfs_v.push_back(sm);
            }
            
        if (!surfs.size()) {
            std::cerr << "ERROR: no seed surfaces found in expansion mode" << std::endl; 
            return EXIT_FAILURE;
        }
        
        std::default_random_engine rng(clock());
        std::shuffle(std::begin(surfs_v), std::end(surfs_v), rng);


        for(auto &it : surfs_v) {
            src = it;
            cv::Mat_<cv::Vec3f> points = src->surface()->rawPoints();
            int w = points.cols;
            int h = points.rows;

            bool found = false;
            for (int r=0;r<10;r++) {
                if ((rand() % 2) == 0)
                {
                    cv::Vec2i p = {rand() % h, rand() % w};
                    
                    if (points(p)[0] != -1 && get_val<double,CachedChunked3dInterpolator<uint8_t,passTroughComputor>>(interpolator, points(p)) >= 128) {
                        found = true;
                        origin = points(p);
                        break;
                    }
                }
                else {
                    cv::Vec2f p;
                    int side = rand() % 4;
                    if (side == 0)
                        p = {rand() % h, 0};
                    else if (side == 1)
                        p = {0, rand() % w};
                    else if (side == 2)
                        p = {rand() % h, w-1};
                    else if (side == 3)
                        p = {h-1, rand() % w};

                    cv::Vec2f searchdir = cv::Vec2f(h/2,w/2) - p;
                    cv::normalize(searchdir, searchdir);
                    found = false;
                    for(int i=0;i<std::min(w/2/abs(searchdir[1]),h/2/abs(searchdir[0]));i++,p+=searchdir) {
                        found = true;
                        cv::Vec2i p_eval = p;
                        for(int r=0;r<5;r++) {
                            cv::Vec2i p_eval = p+r*searchdir;
                            if (points(p_eval)[0] == -1 || get_val<double,CachedChunked3dInterpolator<uint8_t,passTroughComputor>>(interpolator, points(p_eval)) < 128) {
                                found = false;
                                break;
                            }
                        }
                        if (found) {
                            cv::Vec2i p_eval = p+2*searchdir;
                            origin = points(p_eval);
                            break;
                        }
                    }
                }
            }

            if (!found)
                continue;

            count_overlap = 0;
            for(auto comp : surfs_v) {
                if (comp == src)
                    continue;
                if (contains(*comp, origin, search_effort))
                    count_overlap++;
                if (count_overlap >= tgt_overlap_count-1)
                    break;
            }
            if (count_overlap < tgt_overlap_count-1)
                break;
        }

        std::cout << "found potential overlapping starting seed " << origin << " with overlap " << count_overlap << std::endl;
    }
    else if (mode != "gen_neighbor") {
        if (!resume_path.empty()) {
            mode = "resume";
        } else if (use_old_args && argc == 7) {
            mode = "explicit_seed";
            double v;
            interpolator.Evaluate(origin[2], origin[1], origin[0], &v);
            std::cout << "seed location " << origin << " value is " << v << std::endl;
        } else if (!use_old_args && origin[0] != 0 && origin[1] != 0 && origin[2] != 0) {
            mode = "explicit_seed";
            double v;
            interpolator.Evaluate(origin[2], origin[1], origin[0], &v);
            std::cout << "seed location " << origin << " value is " << v << std::endl;
        }
        else {
            mode = "random_seed";
            int count = 0;
            bool succ = false;
            int max_attempts = 1000;
            
            while(count < max_attempts && !succ) {
                origin = {128 + (rand() % (ds->shape(2)-384)), 
                         128 + (rand() % (ds->shape(1)-384)), 
                         128 + (rand() % (ds->shape(0)-384))};

                count++;
                auto chunk_id = chunk_size;
                chunk_id[0] = origin[2]/chunk_id[0];
                chunk_id[1] = origin[1]/chunk_id[1];
                chunk_id[2] = origin[0]/chunk_id[2];

                if (!ds->chunkExists(chunk_id))
                    continue;

                cv::Vec3d dir = {(rand() % 1024) - 512,
                                (rand() % 1024) - 512,
                                (rand() % 1024) - 512};
                cv::normalize(dir, dir);

                for(int i=0;i<128;i++) {
                    double v;
                    cv::Vec3d p = origin + i*dir;
                    interpolator.Evaluate(p[2], p[1], p[0], &v);
                    if (v >= 128) {
                        if (check_existing_segments(tgt_dir, p, name_prefix, search_effort))
                            continue;
                        succ = true;
                        origin = p;
                        std::cout << "Found seed location " << origin << " value: " << v << std::endl;
                        break;
                    }
                }
            }

            if (!succ) {
                std::cout << "ERROR: Could not find valid non-overlapping seed location after " 
                        << max_attempts << " attempts" << std::endl;
                return EXIT_SUCCESS;
            }
        }
    }

    if (thread_limit)
        omp_set_num_threads(thread_limit);

    QuadSurface* resume_surf = nullptr;
    if (mode == "resume") {
        if (corrections.getAllCollections().empty())
           resume_surf = load_quad_from_tifxyz(resume_path);
        else
            resume_surf = load_quad_from_tifxyz(resume_path, SURF_LOAD_IGNORE_MASK);

        origin = {0,0,0}; // Not used in resume mode, but needs to be initialized
    }

    json meta_params;
    meta_params["source"] = "vc_grow_seg_from_seed";
    meta_params["vc_gsfs_params"] = params;
    meta_params["vc_gsfs_mode"] = mode;
    meta_params["vc_gsfs_version"] = "dev";
    if (mode == "expansion")
        meta_params["seed_overlap"] = count_overlap;

    std::string uuid = name_prefix + time_str();
    std::filesystem::path seg_dir = tgt_dir / uuid;

    //
    // gen_neighbor mode: project a source tifxyz surface "in" or "out" along its vertex normals
    // to the first nonzero voxel within a max distance, producing a new tifxyz surface.
    //
    if (mode == "gen_neighbor") {
        if (resume_path.empty()) {
            std::cerr << "ERROR: gen_neighbor mode requires --resume <tifxyz_dir> to provide the source surface" << std::endl;
            return EXIT_FAILURE;
        }

        // Load source surface
        std::unique_ptr<QuadSurface> src_surface;
        try {
            src_surface.reset(load_quad_from_tifxyz(resume_path));
        } catch (const std::exception& ex) {
            std::cerr << "ERROR: failed to load resume surface: " << ex.what() << std::endl;
            return EXIT_FAILURE;
        }

        const cv::Mat_<cv::Vec3f> src_points = src_surface->rawPoints();
        cv::Mat_<cv::Vec3f> dst_points = src_points.clone(); // default fallback: keep original surface
        cv::Mat_<cv::Vec3f> new_points(src_points.size());
        new_points.setTo(cv::Vec3f(-1.f, -1.f, -1.f));

        // Parameters controlling neighbor generation
        const std::string neighbor_dir = params.value("neighbor_dir", std::string("out")); // "in" or "out"
        const double neighbor_step = std::max(1e-3, params.value("neighbor_step", 1.0));   // in voxels
        const double neighbor_max_distance = std::max(0.0, params.value("neighbor_max_distance", 250.0));
        const double neighbor_threshold = params.value("neighbor_threshold", 1.0);          // nonzero by default
        const double neighbor_exit_threshold = std::min(
            params.value("neighbor_exit_threshold", neighbor_threshold * 0.5),
            neighbor_threshold);
        const int neighbor_exit_count = std::max(1, params.value("neighbor_exit_count", 1));
        const double neighbor_min_clearance = std::max(0.0, params.value("neighbor_min_clearance", 0.0));
        const int neighbor_min_clearance_steps =
            std::max(0, params.value("neighbor_min_clearance_steps", 0));

        const int required_clearance_steps = std::max(
            neighbor_min_clearance_steps,
            (neighbor_min_clearance > 0.0)
                ? static_cast<int>(std::ceil(neighbor_min_clearance / neighbor_step))
                : 0);
        const bool neighbor_fill = params.value("neighbor_fill", true);

        const bool cast_out = (neighbor_dir == "out");
        if (!(cast_out || neighbor_dir == "in")) {
            std::cerr << "WARNING: neighbor_dir must be 'in' or 'out'; defaulting to 'out'" << std::endl;
        }

        // Cast a ray per valid vertex
        const int rows = src_points.rows;
        const int cols = src_points.cols;
        const int max_steps = (neighbor_max_distance > 0.0)
                                ? static_cast<int>(std::ceil(neighbor_max_distance / neighbor_step))
                                : 0;

        // Helper to test if a point is a valid vertex (convention: x==-1 denotes invalid)
        auto is_valid_vertex = [](const cv::Vec3f& p) -> bool {
            return p[0] != -1.f && p[1] != -1.f && p[2] != -1.f;
        };

        // Traverse grid
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                const cv::Vec3f start = src_points(r, c);
                if (!is_valid_vertex(start)) {
                    continue;
                }

                // Compute grid normal at this grid location (c=x, r=y)
                cv::Vec3f n = grid_normal(src_points, cv::Vec3f(static_cast<float>(c), static_cast<float>(r), 0.0f));
                if (!std::isfinite(n[0]) || !std::isfinite(n[1]) || !std::isfinite(n[2])) {
                    continue; // leave invalid
                }
                if (!cast_out) {
                    n *= -1.f; // cast "in"
                }
                // Normalize direction defensively
                cv::normalize(n, n);

                // March along the ray until threshold is hit or max distance reached
                bool placed = false;
                bool clearance_met = (required_clearance_steps == 0);
                bool left_surface = false;
                int below_counter = 0;

                for (int k = 1; k <= max_steps; ++k) {
                    const double t = neighbor_step * static_cast<double>(k);
                    if (!clearance_met && k >= required_clearance_steps) {
                        clearance_met = true;
                    }

                    const cv::Vec3d pos = cv::Vec3d(start[0], start[1], start[2]) + cv::Vec3d(n[0], n[1], n[2]) * t;
                    const float v = get_val<double, CachedChunked3dInterpolator<uint8_t,passTroughComputor>>(interpolator, pos);
                    if (!std::isfinite(v)) {
                        continue;
                    }

                    if (!clearance_met) {
                        continue;
                    }

                    if (!left_surface) {
                        if (v <= neighbor_exit_threshold) {
                            below_counter += 1;
                            if (below_counter >= neighbor_exit_count) {
                                left_surface = true;
                            }
                        } else {
                            below_counter = 0;
                        }
                        continue;
                    }

                    if (v >= neighbor_threshold) {
                        new_points(r, c) = cv::Vec3f(static_cast<float>(pos[0]), static_cast<float>(pos[1]), static_cast<float>(pos[2]));
                        placed = true;
                        break;
                    }
                }
                // If not placed, leave invalid (-1,-1,-1)
                (void)placed; // silences unused warning in some builds
            }
        }

        // If enabled, fill holes by averaging neighbors on the new grid
        auto is_valid_vertex_f = [](const cv::Vec3f& p) -> bool {
            return p[0] != -1.f && p[1] != -1.f && p[2] != -1.f;
        };
        if (neighbor_fill) {
            cv::Mat_<cv::Vec3f> cur = new_points.clone();
            cv::Mat_<cv::Vec3f> next = cur.clone();
            const int max_iters = rows + cols; // generous upper bound; exits early when stable
            bool changed = true;
            int iter = 0;
            while (changed && iter < max_iters) {
                int changes = 0;
                // For each invalid cell, average valid 8-neighbors
                #pragma omp parallel for schedule(static) reduction(+:changes)
                for (int r = 0; r < rows; ++r) {
                    for (int c = 0; c < cols; ++c) {
                        if (is_valid_vertex_f(cur(r, c))) {
                            continue;
                        }
                        cv::Vec3d acc(0.0, 0.0, 0.0);
                        int cnt = 0;
                        for (int dr = -1; dr <= 1; ++dr) {
                            for (int dc = -1; dc <= 1; ++dc) {
                                if (dr == 0 && dc == 0) continue;
                                const int rr = r + dr;
                                const int cc = c + dc;
                                if (rr < 0 || cc < 0 || rr >= rows || cc >= cols) continue;
                                const cv::Vec3f& q = cur(rr, cc);
                                if (!is_valid_vertex_f(q)) continue;
                                acc[0] += q[0]; acc[1] += q[1]; acc[2] += q[2];
                                ++cnt;
                            }
                        }
                        if (cnt > 0) {
                            cv::Vec3f avg(static_cast<float>(acc[0] / cnt),
                                          static_cast<float>(acc[1] / cnt),
                                          static_cast<float>(acc[2] / cnt));
                            next(r, c) = avg;
                            changes += 1;
                        } else {
                            next(r, c) = cur(r, c);
                        }
                    }
                }
                changed = (changes > 0);
                std::swap(cur, next);
                ++iter;
            }
            new_points = cur;
        }

        // Merge filled/new points back into destination grid, keeping original where no hit occurred
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                const cv::Vec3f np = new_points(r, c);
                if (is_valid_vertex(np)) {
                    dst_points(r, c) = np;
                }
            }
        }

        // Prepare output surface and save
        std::unique_ptr<QuadSurface> out_surf(new QuadSurface(dst_points, src_surface->scale()));
        // Prepare naming
        std::string neighbor_prefix = std::string("neighbor_") + (cast_out ? "out_" : "in_");
        std::string uuid_local = neighbor_prefix + time_str();
        std::filesystem::path out_dir = tgt_dir / uuid_local;

        // Meta
        nlohmann::json neighbor_meta;
        neighbor_meta["source"] = "vc_grow_seg_from_seed";
        neighbor_meta["vc_gsfs_params"] = params;
        neighbor_meta["vc_gsfs_mode"] = mode;
        neighbor_meta["vc_gsfs_version"] = "dev";

        out_surf->meta = new nlohmann::json(std::move(neighbor_meta));
        out_surf->save(out_dir, uuid_local, true);

        // Done
        return EXIT_SUCCESS;
    }

    QuadSurface *surf = tracer(ds.get(), 1.0, &chunk_cache, origin, params, cache_root, voxelsize, direction_fields, resume_surf, seg_dir, meta_params, corrections);
 
    if (resume_surf) {
        delete resume_surf;
    }

    double area_cm2 = (*surf->meta)["area_cm2"].get<double>();
    if (area_cm2 < min_area_cm) {
        if (std::filesystem::exists(seg_dir)) {
            std::filesystem::remove_all(seg_dir);
        }
        return EXIT_SUCCESS;
    }

    std::cout << "saving " << seg_dir << std::endl;
    surf->save(seg_dir, uuid, true);

    SurfaceMeta current;

    if (mode == "expansion" && !skip_overlap_check) {
        current.path = seg_dir;
        current.setSurface(surf);
        current.bbox = surf->bbox();

        // Read existing overlapping data
        std::set<std::string> current_overlapping = read_overlapping_json(current.path);

        // Add the source segment
        current_overlapping.insert(src->name());

        // Update source's overlapping data
        std::set<std::string> src_overlapping = read_overlapping_json(src->path);
        src_overlapping.insert(current.name());
        write_overlapping_json(src->path, src_overlapping);

        // Check overlaps with existing surfaces
        for(auto &s : surfs_v)
            if (overlap(current, *s, search_effort)) {
                current_overlapping.insert(s->name());

                std::set<std::string> s_overlapping = read_overlapping_json(s->path);
                s_overlapping.insert(current.name());
                write_overlapping_json(s->path, s_overlapping);
            }

        // Check for additional surfaces in target directory
        for (const auto& entry : std::filesystem::directory_iterator(tgt_dir))
            if (std::filesystem::is_directory(entry) && !surfs.count(entry.path().filename()))
            {
                std::string name = entry.path().filename();
                if (name.compare(0, name_prefix.size(), name_prefix))
                    continue;

                if (name == current.name())
                    continue;

                std::filesystem::path meta_fn = entry.path() / "meta.json";
                if (!std::filesystem::exists(meta_fn))
                    continue;

                std::ifstream meta_f(meta_fn);
                json meta = json::parse(meta_f);

                if (!meta.count("bbox"))
                    continue;

                if (meta.value("format","NONE") != "tifxyz")
                    continue;

                SurfaceMeta other = SurfaceMeta(entry.path(), meta);
                other.readOverlapping();

                if (overlap(current, other, search_effort)) {
                    current_overlapping.insert(other.name());

                    std::set<std::string> other_overlapping = read_overlapping_json(other.path);
                    other_overlapping.insert(current.name());
                    write_overlapping_json(other.path, other_overlapping);
                }
            }

        // Write final overlapping data for current
        write_overlapping_json(current.path, current_overlapping);
    }

    delete surf;
    for (auto sm : surfs_v) {
        delete sm;
    }

    return EXIT_SUCCESS;
}
