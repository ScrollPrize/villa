#define VESUVIUS_IMPL
#include "vesuvius-c.h"
#include <stdio.h>

#define TEST_CACHEDIR "./54keV_7.91um_Scroll1A.zarr/0/"
#define TEST_ZARR_URL "https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes_zarr_standardized/54keV_7.91um_Scroll1A.zarr/0/"


int main() {

    int x = 3693, y = 2881, z = 6777;
    int volstart[3] = {z & ~127, y & ~127, x & ~127};
    int chunksize[3] = {256,256,256};
    volume* vol = vs_vol_new(TEST_CACHEDIR, TEST_ZARR_URL);
    chunk* mychunk = vs_vol_get_chunk(vol,volstart,chunksize);

    //get a chunk. If it has not been downloaded, it will be downloaded and saved to the cache
    if (mychunk) {
        printf("Got a chunk: %d+%d, %d+%d, %d+%d\n", volstart[0],mychunk->dims[0],volstart[1],mychunk->dims[1],volstart[2],mychunk->dims[2]);
    }

    // Read a single value from the scroll volume
    unsigned char value = vs_chunk_get(mychunk, z % 256, y % 256, x % 256);

    printf("Voxel value at (%d, %d, %d): %u\n", x, y, z, value);

    //get the same chunk. This time we will read from the on disk cache
    chunk* mychunk2 = vs_vol_get_chunk(vol,volstart,chunksize);

    if (mychunk2) {
        printf("Got a chunk: %d+%d, %d+%d, %d+%d\n", volstart[0],mychunk2->dims[0],volstart[1],mychunk2->dims[1],volstart[2],mychunk2->dims[2]);
    }
    vs_chunk_free(mychunk2);
    mychunk2 = NULL;

    //grab a slice from the middle of the chunk
    slice* myslice = vs_slice_extract(mychunk,mychunk->dims[0]/2);
    vs_bmp_write("xy_slice.bmp",myslice);
    vs_slice_free(myslice);

    chunk* xzchunk = vs_transpose(mychunk,"zyx","yxz");

    slice* xzslice = vs_slice_extract(xzchunk,xzchunk->dims[0]/2);
    vs_bmp_write("xz_slice.bmp",xzslice);
    vs_slice_free(xzslice);
    vs_chunk_free(xzchunk);

    chunk* yzchunk = vs_transpose(mychunk,"zyx","xyz");

    slice* yzslice = vs_slice_extract(yzchunk,yzchunk->dims[0]/2);
    vs_bmp_write("yz_slice.bmp",yzslice);
    vs_slice_free(yzslice);
    vs_chunk_free(yzchunk);

    // Fetch an .obj
    char* buf = NULL;
    long len = vs_download("https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/paths/20231016151002/20231016151002.obj", &buf);
    if (len <= 0) {
        LOG_ERROR("failed to fetch obj");
        return 1;
    }
    FILE* meshfp = fopen("20231016151002.obj", "wb");
    if (meshfp == NULL) {
        return 1;
    }
    fwrite(buf,1,len,meshfp);

    f32* vertices;
    s32* indices;
    s32 vertex_count,index_count;
    if (vs_read_obj("20231016151002.obj",&vertices,&indices,&vertex_count,&index_count)) {
        LOG_ERROR("failed to read obj");
        return 1;
    }

    // Write the triangle mesh to an .obj file
    if (vs_write_obj("20231016151002_dupe.obj",vertices,indices,vertex_count,index_count)) {
        LOG_ERROR("failed to write mesh");
        return 1;
    }


    mesh* mymesh = vs_mesh_new(vertices,NULL,indices,NULL,vertex_count,index_count);
    if (mymesh == NULL) {

    }
    printf("Fetched triangle mesh with %d vertices and %d indices\n", mymesh->vertex_count, mymesh->index_count);

    // Calculate the bounding box of the triangle mesh
    f32 zorigin,yorigin,xorigin,zlength,ylength,xlength;
    vs_mesh_get_bounds(mymesh,&zorigin,&yorigin,&xorigin,&zlength,&ylength,&xlength);
    printf("Bounding box of the triangle mesh: %f+%f, %f+%f, %f+%f\n", zorigin,zlength,yorigin,ylength,xorigin,xlength);

    vs_mesh_translate(mymesh,1.0f,1.0f,1.0f);

    return 0;
}
