#include <nlohmann/json.hpp>

#include "vc/core/util/xtensor_include.hpp"
#include XTENSORINCLUDE(containers, xarray.hpp)
#include XTENSORINCLUDE(io, xio.hpp)
#include XTENSORINCLUDE(views, xview.hpp)

#include "z5/factory.hxx"
#include "z5/filesystem/handle.hxx"
#include "z5/multiarray/xtensor_access.hxx"
#include "z5/attributes.hxx"

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/StreamOperators.hpp"


#include <filesystem>



cv::Vec3f parse_vec3f(std::string line, std::string type = "")
{
    cv::Vec3f v;
    std::istringstream iss(line);
    std::string t;
    if (!(iss >> t >> v[0] >> v[1] >> v[2]) || (type.size() && t != type)) {
        std::cout << t << v << type << line << std::endl;
        throw std::runtime_error("error in parse_vec3f()");
    }
    return v;
}

bool istype(const std::string &line, const std::string &type)
{
    return line.rfind(type+" ", 0) == 0;
}


struct DSReader
{
    z5::Dataset *ds;
    float scale;
    ChunkCache *cache;
};

float alphacomp_offset(DSReader &reader, cv::Vec3f point, cv::Vec3f normal, float start, float stop, float step, float low, float high, int r)
{
    int d = 2*r+1;
    cv::Size size = {d,d};
    cv::Point2i c = {r,r};

    float transparent = 1;
    cv::Mat_<float> blur(size, 0);
    float integ_z = 0;

    cv::Mat_<cv::Vec3f> coords;
    PlaneSurface plane(point, normal);
    plane.gen(&coords, nullptr, size, cv::Vec3f(0,0,0), reader.scale, {0,0,0});

    coords *= reader.scale;
    float s = copysignf(1.0,step);

    for(double off=start;off*s<=stop*s;off+=step) {
        cv::Mat_<uint8_t> slice;
        //I hate opencv
        cv::Mat_<cv::Vec3f> offmat(size, normal*off*reader.scale);
        readInterpolated3D(slice, reader.ds, coords+offmat, reader.cache);

        cv::Mat floatslice;
        slice.convertTo(floatslice, CV_32F, 1/255.0);

        cv::GaussianBlur(floatslice, blur, {d,d}, 0);
        cv::Mat_<float> opaq_slice = blur;

        opaq_slice = (opaq_slice-low)/(high-low);
        opaq_slice = cv::min(opaq_slice,1);
        opaq_slice = cv::max(opaq_slice,0);

        float joint = transparent*opaq_slice(c);
        integ_z += joint * off;
        transparent = transparent-joint;
    }

    integ_z /= (1-transparent+1e-5);

    return integ_z;
}

int process_obj(const std::string &src, const std::string &tgt, DSReader &reader, const nlohmann::json &params)
{
    std::ifstream obj(src);
    std::ofstream out(tgt);
    std::string line;
    std::string last_line;
    int v_count = 0;
    int vn_count = 0;
    std::vector<cv::Vec3f> vs;
    std::vector<cv::Vec3f> vns;

    while (std::getline(obj, line))
    {
        if (istype(line, "v")) {
            if (vs.size() != vns.size())
                throw std::runtime_error("sorry our taste in obj is quite peculiar ...");
            vs.push_back(parse_vec3f(line));
        }
        if (istype(line, "vn")) {
            if (vs.size()-1 != vns.size())
                throw std::runtime_error("sorry our taste in obj is quite peculiar ...");
                cv::Vec3f normal = parse_vec3f(line);
                normalize(normal, normal);
                vns.push_back(normal);
        }
        // if (vs.size() % 10000 == 0)
            // std::cout << vs.size() << std::endl;
    }

    if (vs.size() != vns.size())
        throw std::runtime_error("sorry our taste in obj is quite peculiar ...");

    cv::Mat_<cv::Vec3f> vs_mat(vs.size(), 1, &vs[0]);
    cv::Mat_<cv::Vec3f> vns_mat(vs.size(), 1, &vns[0]);
    cv::Mat_<uint8_t> slice;

    if (params.count("refine") && params.at("refine").get<bool>()) {
        float start = params.value<float>("start", -6.0);
        float stop = params.value<float>("stop", 30.0);
        float step = params.value<float>("step", 2.0);
        float low = params.value<float>("low", 0.1);
        float high = params.value<float>("high", 1.0);
        float border_off = params.value<float>("border_off", 1.0);
        int r = params.value<int>("r", 3);

#pragma omp parallel for
        for(int j=0;j<vs.size();j++) {
            float off = alphacomp_offset(reader, vs[j], vns[j], start, stop, step, low, high, r);
            vs[j] += vns[j]*(off + border_off);
        }
    }

    bool vertexcolor = false;
    if (params.count("gen_vertexcolor") && params.at("gen_vertexcolor").get<bool>()) {
        vertexcolor = true;
        readInterpolated3D(slice, reader.ds, vs_mat*reader.scale, reader.cache);
    }

    obj.clear();
    obj.seekg(0);
    int v_counter = 0;
    while (std::getline(obj, line))
    {
        if (istype(line, "v")) {
            cv::Vec3f v = vs[v_counter];
            out << "v " << v[0] << " " << v[1] << " " << v[2];
            if (vertexcolor) {
                float col = int(slice(v_counter, 0)*1000/255.0)/1000.0;
                out << " " << col << " " << col << " " << col;
            }
            out << "\n";
            v_counter++;
        }
        else
            out << line << "\n";
    }

    return EXIT_SUCCESS;
}

class MeasureLife
{
public:
    MeasureLife(std::string msg)
    {
        std::cout << msg << std::flush;
        start = std::chrono::high_resolution_clock::now();
    }
    ~MeasureLife()
    {
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << " took " << std::chrono::duration<double>(end-start).count() << " s" << std::endl;
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
};

int main(int argc, char *argv[])
{
    if (argc != 5) {
        std::cout << "usage: " << argv[0] << " <zarr-volume> <src-obj> <out-obj> <json-params>" << std::endl;
        return EXIT_SUCCESS;
    }

    const char *vol_path = argv[1];
    const char *obj_src = argv[2];
    const char *obj_tgt = argv[3];
    const char *params_path = argv[4];

    std::ifstream params_f(params_path);
    nlohmann::json params = nlohmann::json::parse(params_f);
  
    z5::filesystem::handle::Group group(vol_path, z5::FileMode::FileMode::r);
    z5::filesystem::handle::Dataset ds_handle(group, "1", "/");
    std::unique_ptr<z5::Dataset> ds = z5::filesystem::openDataset(ds_handle);

    std::cout << "zarr dataset size for scale group 1 " << ds->shape() << std::endl;
    std::cout << "chunk shape shape " << ds->chunking().blockShape() << std::endl;
    
    ChunkCache chunk_cache(10e9);

    DSReader reader = {ds.get(), 0.5, &chunk_cache};

    MeasureLife *timer = new MeasureLife("processing obj ...\n");
    int ret = process_obj(obj_src, obj_tgt, reader, params);
    delete timer;
    return ret;
}
