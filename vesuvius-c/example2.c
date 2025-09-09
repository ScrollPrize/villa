#define VESUVIUS_IMPL

#include "vesuvius-c.h"

int test_volume_load() {
    const char* zarray_url = "https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes_zarr_standardized/54keV_7.91um_Scroll1A.zarr/0/.zarray";
    const char* zarr_block_url = "https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes_zarr_standardized/54keV_7.91um_Scroll1A.zarr/0/50/30/30";
    void* metadata_buf;
    printf("downloading %s\n",zarray_url);
    if (vs_download(zarray_url, &metadata_buf) <= 0) {
        return 1;
    }

    zarr_metadata metadata;
    printf("parsing zarr metadata\n");
    if (vs_zarr_parse_metadata(metadata_buf, &metadata)) {
        return 1;
    }

    void* compressed_buf;
    long compressed_size;
    printf("downloading %s\n",zarr_block_url);
    if ((compressed_size = vs_download(zarr_block_url, &compressed_buf)) <= 0) {
        return 1;
    }
    printf("decompressing zarr chunk\n");
    chunk* mychunk = vs_zarr_decompress_chunk(compressed_size, compressed_buf,metadata);
    if (mychunk == NULL) {
        return 1;
    }
    printf("rescaling zarr chunk\n");
    chunk* rescaled = vs_normalize_chunk(mychunk);
    s32 vertex_count, index_count;
    f32 *vertices;
    s32 *indices;
    f32* graycolors;
    printf("marching cubes on rescaled chunk\n");
    if (vs_march_cubes(rescaled->data,rescaled->dims[0],rescaled->dims[1],rescaled->dims[2],.5f,&vertices,&graycolors,&indices,&vertex_count,&index_count)) {
        return 1;
    }
    printf("writing mesh to out_vol.ply\n");
    if (vs_ply_write("out_vol.ply",vertices,NULL,NULL,indices,vertex_count,index_count)) {
        return 1;
    }

    return 0;
}

int test_fiber_load() {
    const char* zarray_url = "https://dl.ash2txt.org/community-uploads/bruniss/Fiber-and-Surface-Models/GP-Predictions/updated_zarrs/mask-2ext-surface_ome.zarr/0/.zarray";
    const char* zarr_block_url = "https://dl.ash2txt.org/community-uploads/bruniss/Fiber-and-Surface-Models/GP-Predictions/updated_zarrs/mask-2ext-surface_ome.zarr/0/50/30/30";

    chunk* mychunk;

    void* metadata_buf;
    printf("downloading %s\n",zarray_url);
    if (vs_download(zarray_url, &metadata_buf) <= 0) {
        return 1;
    }
    zarr_metadata metadata;
    printf("parsing zarray metadata\n");
    if (vs_zarr_parse_metadata(metadata_buf, &metadata)) {
        return 1;
    }
    void* compressed_buf;
    long compressed_size;
    printf("downloading %s\n",zarr_block_url);
    if ((compressed_size = vs_download(zarr_block_url, &compressed_buf)) <= 0) {
        return 1;
    }
    printf("decompressing zarr chunk\n");
    mychunk = vs_zarr_decompress_chunk(compressed_size, compressed_buf,metadata);
    if (mychunk == NULL) {
        return 1;
    }
    printf("rescaling zarr chunk\n");
    chunk* rescaled = vs_normalize_chunk(mychunk);
    s32 vertex_count, index_count;
    f32 *vertices;
    f32 *graycolors;
    s32 *indices;
    printf("marching cubes on rescaled chunk\n");
    if (vs_march_cubes(rescaled->data,rescaled->dims[0],rescaled->dims[1],rescaled->dims[2],.5f,&vertices,&graycolors,&indices,&vertex_count,&index_count)) {
        return 1;
    }
    printf("writing mesh to out_surface.ply\n");
    if (vs_ply_write("out_surface.ply",vertices,NULL,NULL,indices,vertex_count,index_count)) {
        return 1;
    }

    return 0;
}

void test_christmas_highlighting() {
    int vol_start[3] = {3072,3072,3072};
    int chunk_dims[3] = {1024,256,256};
     volume* scroll_vol = vs_vol_new("./54keV_7.91um_Scroll1A.zarr/0/",
     "https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes_zarr_standardized/54keV_7.91um_Scroll1A.zarr/0/");
  chunk* scroll_chunk = vs_vol_get_chunk(scroll_vol, vol_start,chunk_dims);

  volume* fiber_vol = vs_vol_new("s1-surface-regular.zarr/",
     "https://dl.ash2txt.org/community-uploads/bruniss/Fiber-and-Surface-Models/Predictions/s1/full-scroll-preds/s1-surface-regular.zarr/");

  chunk* fiber_chunk = vs_vol_get_chunk(fiber_vol, vol_start,chunk_dims);

  chunk* rescaled_vol = vs_normalize_chunk(scroll_chunk);
  chunk* rescaled_fiber = vs_normalize_chunk(fiber_chunk);

  chunk* out_r = vs_chunk_new(chunk_dims);
  chunk* out_g = vs_chunk_new(chunk_dims);
  chunk* out_b = vs_chunk_new(chunk_dims);

   f32 ISO = 0.2f;

  for (int z = 0; z < chunk_dims[0]-1; z++) {
    for (int y = 0; y < chunk_dims[1]; y++) {
      for (int x = 0; x < chunk_dims[2]; x++) {
        f32 me   = vs_chunk_get(rescaled_vol,z,y,x);
        f32 next = vs_chunk_get(rescaled_vol,z+1,y,x);
        f32 fiber = vs_chunk_get(rescaled_fiber,z,y,x);

        if (fiber > 0) {
          vs_chunk_set(out_b,z,y,x,1.0f);
        } else {
          vs_chunk_set(out_b,z,y,x,0.0f);
        }
        if (me > ISO && next > ISO) {
          //scroll gray
          if (vs_chunk_get(out_b,z,y,x) <= 0.0001f) {
            vs_chunk_set(out_r,z,y,x,me);
            vs_chunk_set(out_g,z,y,x,me);
            vs_chunk_set(out_b,z,y,x,me);
          }
        } else if (me < ISO && next < ISO) {
          if (vs_chunk_get(out_b,z,y,x) <= 0.0001f) {
            vs_chunk_set(out_r,z,y,x,0.0f);
            vs_chunk_set(out_g,z,y,x,0.0f);
            vs_chunk_set(out_b,z,y,x,0.0f);
          }
          //black
        } else if (me < ISO && next >= ISO) {
          if (vs_chunk_get(out_b,z,y,x) <= 0.0001f) {
            vs_chunk_set(out_r,z,y,x,1.0f);
            vs_chunk_set(out_g,z,y,x,0.0f);
            //vs_chunk_set(out_b,z,y,x,0.0f);
          }
          //red
        } else if (me >= ISO && next < ISO) {
          if (vs_chunk_get(out_b,z,y,x) <= 0.0001f) {
            vs_chunk_set(out_r,z,y,x,0.0f);
            vs_chunk_set(out_g,z,y,x,1.0f);
            //vs_chunk_set(out_b,z,y,x,0.0f);
          }
        //green
        }
      }
    }
  }

  vs_chunks_to_video(out_r, out_g, out_b, "output.mp4", 30);
}

int main(int argc, char** argv) {
  if (test_volume_load()) {
    return 1;
  }
  if (test_fiber_load()) {
    return 1;
  }
    test_christmas_highlighting();
  return 0;
}
