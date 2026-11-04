#pragma once
#include "transfer.hpp"
#include "mask.hpp"
namespace vcr::projection::python {
inline void bind(nb::module_& m){
    m.def("_project_mask",[](geometry::Array positions,geometry::Array normals,geometry::Array mask,geometry::Array valid,std::array<std::int64_t,3> origin,float thickness,geometry::Array output){
        if(positions.ndim()!=3||output.ndim()!=3)throw nb::value_error("expected H,W,3 positions and 3D output");
        int id=geometry::array_device(positions);auto h=positions.shape(0),w=positions.shape(1);
        geometry::check(positions,nb::dtype<float>(),{h,w,3},id);geometry::check(normals,nb::dtype<float>(),{h,w,3},id);
        for(auto a:{mask,valid})geometry::check(a,nb::dtype<bool>(),{h,w},id);
        geometry::check(output,nb::dtype<std::uint32_t>(),{output.shape(0),output.shape(1),output.shape(2)},id);
        geometry::disjoint({positions,normals,mask,valid},{output});MaskBand g;g.rows=int(h);g.cols=int(w);
        g.positions=static_cast<const float*>(positions.data());g.normals=static_cast<const float*>(normals.data());g.mask=static_cast<const bool*>(mask.data());g.valid=static_cast<const bool*>(valid.data());g.halfThickness=thickness;
        for(int a=0;a<3;++a){g.origin[a]=origin[a];g.shape[a]=output.shape(a);}
        nb::gil_scoped_release release;projectMask(g,static_cast<std::uint32_t*>(output.data()),id);
    });
    m.def("_surface_transfer",[](std::vector<geometry::Array> a,std::array<int,2> label,std::array<int,2> output,std::array<int,4> bounds,std::array<float,2> offset,float distance,float seam,geometry::Array indices,geometry::Array valid,geometry::Array distances){
        using geometry::check;using geometry::array_device;
        if(a.size()!=12||a[0].ndim()!=4||a[3].ndim()!=4)throw nb::value_error("invalid surface transfer arrays");
        int id=array_device(a[0]);auto h=a[0].shape(1),w=a[0].shape(2),sh=a[3].shape(1),sw=a[3].shape(2);
        check(a[0],nb::dtype<float>(),{1,h,w,3},id);check(a[1],nb::dtype<bool>(),{1,h,w},id);check(a[2],nb::dtype<float>(),{1,3,h,w,3},id);
        check(a[3],nb::dtype<float>(),{1,sh,sw,3},id);check(a[4],nb::dtype<bool>(),{1,sh,sw},id);
        for(int k:{5,6})check(a[k],nb::dtype<float>(),{h,w},id);check(a[7],nb::dtype<bool>(),{h,w},id);
        for(int k:{8,9,10}){if(a[k].size())check(a[k],nb::dtype<float>(),{h,w},id);}
        if(bool(a[8].size())!=bool(a[9].size())||(a[10].size()&&(!a[8].size()||!std::isfinite(seam)||seam<=0)))throw nb::value_error("invalid seam fields");
        if(a[11].size())check(a[11],nb::dtype<std::uint8_t>(),{std::size_t(label[0]),std::size_t(label[1])},id);
        if(bounds[1]<=bounds[0]||bounds[3]<=bounds[2])throw nb::value_error("empty tile");
        auto th=std::size_t(bounds[1]-bounds[0]),tw=std::size_t(bounds[3]-bounds[2]);
        check(indices,nb::dtype<std::int64_t>(),{th,tw},id);check(valid,nb::dtype<std::uint8_t>(),{th,tw},id);check(distances,nb::dtype<float>(),{th,tw},id);
        for(auto input:a)geometry::disjoint({input},{indices,valid,distances});
        RasterRequest r{};r.target_height=h;r.target_width=w;r.source_height=sh;r.source_width=sw;r.label_height=label[0];r.label_width=label[1];r.output_height=output[0];r.output_width=output[1];
        r.row_start=bounds[0];r.row_end=bounds[1];r.col_start=bounds[2];r.col_end=bounds[3];r.label_offset_y=offset[0];r.label_offset_x=offset[1];r.max_distance=distance;r.max_seam_distance=seam;
        auto f=[&](int k){return static_cast<const float*>(a[k].data());};auto u=[&](int k){return static_cast<const std::uint8_t*>(a[k].data());};
        r.target.xyz=f(0);r.target.valid=static_cast<const bool*>(a[1].data());r.target.du=f(2);r.target.dv=f(2)+h*w*3;r.target.duv=f(2)+h*w*6;
        r.source.xyz=f(3);r.source.valid=static_cast<const bool*>(a[4].data());r.source.triangles=true;
        r.uv_rows=f(5);r.uv_cols=f(6);r.uv_valid=u(7);
        r.fill_seams=bool(a[8].size());r.has_seam_limit=bool(a[10].size());r.has_source_validity=bool(a[11].size());
        r.filled_uv_rows=f(8);r.filled_uv_cols=f(9);r.filled_uv_anchor_distance=f(10);r.source_validity=u(11);
        r.output_source_indices=static_cast<std::int64_t*>(indices.data());r.output_validity=static_cast<std::uint8_t*>(valid.data());r.output_distances=static_cast<float*>(distances.data());
        RasterResult result{};{nb::gil_scoped_release release;transfer(r,result,id);}
        return std::array<std::int64_t,5>{result.target_surface_valid,result.measured_pixels,result.seam_filled_pixels,result.seam_blocked_pixels,result.inherited_filled_pixels};
    });
}
}
