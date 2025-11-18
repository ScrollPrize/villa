#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfaceModeling.hpp"
#include "vc/core/types/ChunkedTensor.hpp"

#include "z5/factory.hxx"
#include <nlohmann/json.hpp>

#include <opencv2/highgui.hpp>
#include <omp.h>



using json = nlohmann::json;


static inline float sdist(const cv::Vec3f &a, const cv::Vec3f &b)
{
    cv::Vec3f d = a-b;
    // return d.dot(d);
    return d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
}

//min-loc that allows going off-surface and does _not_ avoid holes (which could produce artifacts)
float min_loc_plain(const cv::Mat_<cv::Vec3f> &points, cv::Vec2f &loc, cv::Vec3f &out, const cv::Vec3f &tgt, float init_step, float min_step)
{
    if (!loc_valid(points, {loc[1],loc[0]})) {
        out = {-1,-1,-1};
        return -1;
    }
    
    cv::Rect bounds = {0,0,points.cols-1,points.rows-1};
    
    bool changed = true;
    cv::Vec3f val = at_int(points, loc);
    out = val;
    float best = sdist(val, tgt);
    float res;
    
    // std::vector<cv::Vec2f> search = {{0,-1},{0,1},{-1,-1},{-1,0},{-1,1},{1,-1},{1,0},{1,1}};
    std::vector<cv::Vec2f> search = {{0,-1},{0,1},{-1,0},{1,0}};
    float step = init_step;
    
    while (changed) {
        changed = false;
        
        for(auto &off : search) {
            cv::Vec2f cand = loc+off*step;
            
            if (!bounds.contains(cv::Point(cand[0],cand[1])))
                continue;
            
            val = at_int(points, cand);
            res = sdist(val, tgt);
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
        
        if (step < min_step)
            break;
    }

    return best;
}

float find_loc_wind_slow(cv::Vec2f &loc, float tgt_wind, const cv::Mat_<cv::Vec3f> &points, const cv::Mat_<float> &winding, const cv::Vec3f &tgt, float th)
{
    float best_res = -1;
    uint32_t sr = loc[0]+loc[1];
    for(int r=0;r<1000;r++) {
        cv::Vec2f cand = loc;
        
        if (r)
            cand = {static_cast<float>(rand_r(&sr) % points.cols), static_cast<float>(rand_r(&sr) % points.rows)};
        
        if (std::isnan(winding(cand[1],cand[0])) || abs(winding(cand[1],cand[0])-tgt_wind) > 0.5)
            continue;
        
        cv::Vec3f out_;
        float res = min_loc_plain(points, cand, out_, tgt, 4.0, 0.01);
        
        if (res < 0)
            continue;
        
        if (std::isnan(winding(cand[1],cand[0])) || abs(winding(cand[1],cand[0])-tgt_wind) > 0.3)
            continue;
        
        if (res < th) {
            loc = cand;
            return res;
        }
        
        if (best_res == -1 || res < best_res) {
            loc = cand;
            best_res = res;
        }
    }
    
    return sqrt(best_res);
}

bool loc_surround_valid(const cv::Mat_<cv::Vec3f> &m, const cv::Vec2f &loc)
{
    std::vector<cv::Vec2i> neighs = {{0,0},{0,1},{1,0},{0,-1},{-1,0}};
    
    for(auto n : neighs)
        if (!loc_valid(m, loc+cv::Vec2f(n)))
            return false;

    return true;
}

//FIXME use min-loc that runs into forbidden areas!

cv::Mat_<cv::Vec3f> points_hr_grounding(cv::Mat_<float> wind_lr, const cv::Mat_<cv::Vec3f> &points_lr, const std::vector<cv::Mat_<float>> &wind_hr_src, 
    const std::vector<cv::Mat_<cv::Vec3f>> &points_hr_src, int scale)
{
    cv::Mat_<cv::Vec3f> points_hr(points_lr.rows*scale, points_lr.cols*scale, {0,0,0});
    cv::Mat_<int> counts_hr(points_lr.rows*scale, points_lr.cols*scale, 0);
    
    cv::Mat_<cv::Vec3f> points_lr_grounded(points_lr.size(), 0);
    cv::Mat_<int> counts_lr_grounded(points_lr.size(), 0);

    
    for(int n=0;n<points_hr_src.size();n++) {
        int succ = 0;
        for(int j=0;j<points_lr.rows-1;j++)
            for(int i=0;i<points_lr.cols-1;i++) {
                if (points_lr(j,i)[0] == -1)
                    continue;
                if (points_lr(j,i+1)[0] == -1)
                    continue;
                if (points_lr(j+1,i)[0] == -1)
                    continue;
                if (points_lr(j+1,i+1)[0] == -1)
                    continue;
                
                cv::Vec2f l00, l01, l10, l11;
                
                float hr_th = 20.0;
                float res;
                cv::Vec3f out_;

                res = find_loc_wind_slow(l00, wind_lr(j,i), points_hr_src[n], wind_hr_src[n], points_lr(j,i), hr_th*hr_th);
                if (res < 0 || res > hr_th*hr_th || !loc_valid(points_hr_src[n], {l00[1],l00[0]}))
                    continue;

                l01 = l00;
                res = min_loc_plain(points_hr_src[n], l01, out_, points_lr(j,i+1), 1.0, 0.01);
                if (res < 0 || res > hr_th*hr_th || !loc_valid(points_hr_src[n], {l01[1],l01[0]}))
                    continue;
                l10 = l00;
                res = min_loc_plain(points_hr_src[n], l10, out_, points_lr(j+1,i), 1.0, 0.01);
                if (res < 0 || res > hr_th*hr_th || !loc_valid(points_hr_src[n], {l10[1],l10[0]}))
                    continue;
                l11 = l00;
                res = min_loc_plain(points_hr_src[n], l11, out_, points_lr(j+1,i+1), 1.0, 0.01);
                if (res < 0 || res > hr_th*hr_th || !loc_valid(points_hr_src[n], {l11[1],l11[0]}))
                    continue;
                
                //FIXME should also re-use already found corners for interpolation!
                points_lr_grounded(j, i) += at_int(points_hr_src[n], l00);
                points_lr_grounded(j, i+1) += at_int(points_hr_src[n], l01);
                points_lr_grounded(j+1, i) += at_int(points_hr_src[n], l10);
                points_lr_grounded(j+1, i+1) += at_int(points_hr_src[n], l11);
                counts_lr_grounded(j, i) += 1;
                counts_lr_grounded(j, i+1) += 1;
                counts_lr_grounded(j+1, i) += 1;
                counts_lr_grounded(j+1, i+1) += 1;                
                
                l00 = {l00[1],l00[0]};
                l01 = {l01[1],l01[0]};
                l10 = {l10[1],l10[0]};
                l11 = {l11[1],l11[0]};                
                
                // std::cout << "succ!" << res << cv::Vec2i(i,j) << l00 << l01 << points_tgt(j,i) << std::endl;
                
                for(int sy=0;sy<=scale;sy++)
                    for(int sx=0;sx<=scale;sx++) {
                        float fx = float(sx)/scale;
                        float fy = float(sy)/scale;
                        cv::Vec2f l0 = (1-fx)*l00 + fx*l01;
                        cv::Vec2f l1 = (1-fx)*l10 + fx*l11;
                        cv::Vec2f l = (1-fy)*l0 + fy*l1;
                        
                        if (loc_valid(points_hr_src[n], l)) {
                            succ++;
                            points_hr(j*scale+sy,i*scale+sx) += at_int(points_hr_src[n], {l[1],l[0]});
                            counts_hr(j*scale+sy,i*scale+sx) += 1;
                        }
                    }
            }
        std::cout << "grounded " << succ << std::endl;
    }
    
    for(int j=0;j<points_lr_grounded.rows;j++)
        for(int i=0;i<points_lr_grounded.cols;i++)
            if (counts_lr_grounded(j,i))
                points_lr_grounded(j,i) = points_lr_grounded(j,i) / counts_lr_grounded(j,i);
            else
                points_lr_grounded(j,i) = points_lr(j, i);
            
    cv::Mat_<int> counts_hr_grounded = counts_hr.clone();
            
    for(int n=0;n<points_hr_src.size();n++)
        for(int j=0;j<points_lr_grounded.rows-1;j++)
            for(int i=0;i<points_lr_grounded.cols-1;i++) {
                if (points_lr_grounded(j,i)[0] == -1)
                    continue;
                if (points_lr_grounded(j,i+1)[0] == -1)
                    continue;
                if (points_lr_grounded(j+1,i)[0] == -1)
                    continue;
                if (points_lr_grounded(j+1,i+1)[0] == -1)
                    continue;
                for(int sy=0;sy<=scale;sy++)
                    for(int sx=0;sx<=scale;sx++) {
                        float fx = float(sx)/scale;
                        float fy = float(sy)/scale;
                        cv::Vec3f c0 = (1-fx)*points_lr_grounded(j,i) + fx*points_lr_grounded(j,i+1);
                        cv::Vec3f c1 = (1-fx)*points_lr_grounded(j+1,i) + fx*points_lr_grounded(j+1,i+1);
                        cv::Vec3f c = (1-fy)*c0 + fy*c1;
                        
                        if (!counts_hr_grounded(j*scale+sy,i*scale+sx)) {
                            points_hr(j*scale+sy,i*scale+sx) += c;
                            counts_hr(j*scale+sy,i*scale+sx) += 1;
                        }
                    }
                }

#pragma omp parallel for
    for(int j=0;j<points_hr.rows;j++)
        for(int i=0;i<points_hr.cols;i++)
            if (counts_hr(j,i))
                points_hr(j,i) /= counts_hr(j,i);
    else
        points_hr(j,i) = {-1,-1,-1};
    
    return points_hr;
}

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

int main(int argc, char *argv[])
{
    if (argc < 6 || (argc-4) % 2 != 0)  {
        std::cout << "usage: " << argv[0] << " <tiffxyz-lr> <winding-lr> <scale-factor> <tiffxyz-highres1> <winding1>  ..." << std::endl;
        std::cout << "   upsamples a lr tiffxyz by interpolating locations from multiple hr surfaces each provided as a pair of tiffxyz, winding-tiff" << std::endl;
        return EXIT_SUCCESS;
    }
    
    std::vector<QuadSurface*> surfs;
    std::vector<cv::Mat_<cv::Vec3f>> surf_points;
    std::vector<cv::Mat_<float>> winds;

    QuadSurface *surf_lr = load_quad_from_tifxyz(argv[1]);
    cv::Mat_<cv::Vec3f> points_lr = surf_lr->rawPoints();
    cv::Mat_<float> wind_lr = cv::imread(argv[2], cv::IMREAD_UNCHANGED);
    
    if (points_lr.size() != wind_lr.size())
        throw std::runtime_error("tiffxyz-lr data must have same size as winding-lr");

    int scale_factor = atoi(argv[3]);
    
    for(int j=0;j<wind_lr.rows;j++)
        for(int i=0;i<wind_lr.cols;i++)
            if (points_lr(j,i)[0] == -1)
                wind_lr(j,i) = NAN;
    
    for(int n=0;n<(argc-4)/2;n++) {
        QuadSurface *surf = load_quad_from_tifxyz(argv[n*2+4]);
        
        std::cout << "surf " << argv[n*2+4] << std::endl;
        
        cv::Mat_<float> wind = cv::imread(argv[n*2+5], cv::IMREAD_UNCHANGED);
                    
        cv::Mat_<cv::Vec3f> points = surf->rawPoints();
        
        for(int j=0;j<wind.rows;j++)
            for(int i=0;i<wind.cols;i++)
                if (points(j,i)[0] == -1)
                    wind(j,i) = NAN;
        
        surfs.push_back(surf);
        winds.push_back(wind);
        surf_points.push_back(points);
    }
    
    for(int i=0;i<surfs.size();i++) {
        //try to find random matches between the surfaces, always coming from surf 0 for now
        std::vector<float> offsets;
        std::vector<float> offsets_rev;
        
        for(int r=0;r<1000;r++) {
            cv::Vec2i p = {rand() % wind_lr.rows, rand() % wind_lr.cols};
            if (points_lr(p)[0] == -1)
                continue;
            
            auto ptr = surfs[i]->pointer();
            float res = surfs[i]->pointTo(ptr, points_lr(p), 2.0);
            
            if (res < 0 || res >= 2)
                continue;
            
            cv::Vec3f loc = surfs[i]->loc_raw(ptr);

            offsets.push_back(wind_lr(p) - at_int(winds[i], {loc[0],loc[1]}));
            offsets_rev.push_back(wind_lr(p) + at_int(winds[i], {loc[0],loc[1]}));
        }
        
        std::sort(offsets.begin(), offsets.end());
        std::sort(offsets_rev.begin(), offsets_rev.end());
        std::cout << "off 0.1 " << offsets[offsets.size()*0.1] << std::endl;
        std::cout << "off 0.5 " << offsets[offsets.size()*0.5] << std::endl;
        std::cout << "off 0.9 " << offsets[offsets.size()*0.9] << std::endl;
        float div_fw = std::abs(offsets[offsets.size()*0.9] - offsets[offsets.size()*0.1]);
        
        
        std::cout << "off_rev 0.1 " << offsets_rev[offsets_rev.size()*0.1] << std::endl;
        std::cout << "off_rev 0.5 " << offsets_rev[offsets_rev.size()*0.5] << std::endl;
        std::cout << "off_rev 0.9 " << offsets_rev[offsets_rev.size()*0.9] << std::endl;
        float div_bw = std::abs(offsets_rev[offsets.size()*0.9] - offsets_rev[offsets.size()*0.1]);
        
        
        if (div_fw < div_bw)
            winds[i] += offsets[offsets.size()*0.5];
        else
            winds[i] = -winds[i] + offsets_rev[offsets_rev.size()*0.5];
    }
    
    {
        cv::Mat_<cv::Vec3f> points_hr = points_hr_grounding(wind_lr, points_lr, winds, surf_points, scale_factor);
        QuadSurface *surf_hr = new QuadSurface(points_hr, surfs[0]->_scale);
        std::filesystem::path tgt_dir = "./";
        surf_hr->meta = new nlohmann::json;
        (*surf_hr->meta)["vc_tiffxyz_upscale_grounding_scale_factor"] = scale_factor;
        std::string name_prefix = "grounding_hr_";
        std::string uuid = name_prefix + time_str();
        std::filesystem::path seg_dir = tgt_dir / uuid;
        std::cout << "saving " << seg_dir << std::endl;
        surf_hr->save(seg_dir, uuid);
        delete surf_hr;
    }
    
    return EXIT_SUCCESS;
}
