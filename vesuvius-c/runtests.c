#define VESUVIUS_IMPL
#include "vesuvius-c.h"

#define TEST_CACHEDIR "./54keV_7.91um_Scroll1A.zarr/0/"
#define TEST_ZARR_URL "https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes_zarr_standardized/54keV_7.91um_Scroll1A.zarr/0/"
#define TEST_ZARRAY_URL "https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes_zarr_standardized/54keV_7.91um_Scroll1A.zarr/0/.zarray"
#define TEST_ZARR_BLOCK_URL "https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes_zarr_standardized/54keV_7.91um_Scroll1A.zarr/0/30/30/30"

int testcurl() {
  printf("%s\n",__FUNCTION__);
  int ret = 0;
  char* buf;
  long len = vs_download("https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/paths/20230503225234/author.txt", &buf);
  if(len!= 6) {
    ret = 1; goto cleanup;
  }
  if (strncmp(buf,"noemi",5) != 0) {ret = 1; goto cleanup;}
  cleanup:
  free(buf);
  printf("%s done \n",__FUNCTION__);
  return ret;
}

int testhistogram() {
  printf("%s\n", __FUNCTION__);
  int ret = 0;

  volume* vol = vs_vol_new(TEST_CACHEDIR, TEST_ZARR_URL);
  chunk* mychunk = vs_vol_get_chunk(vol, (s32[3]){2048,2048,2048},(s32[3]){128,128,128});
  if (mychunk == NULL) { ret = 1; goto cleanup; }

  slice* myslice = vs_slice_new((s32[2]){384,384});
  if (myslice == NULL) { ret = 1; goto cleanup; }

  for (int y = 0; y < 384; y++) {
    for (int x = 0; x < 384; x++) {
      vs_slice_set(myslice,y,x,vs_chunk_get(mychunk,0,y,x));
    }
  }

  histogram* slice_hist = vs_slice_histogram(myslice->data, myslice->dims[0], myslice->dims[1], 256);
  if (slice_hist == NULL) { ret = 1; goto cleanup; }
  histogram* chunk_hist = vs_chunk_histogram(mychunk->data, mychunk->dims[0], mychunk->dims[1], mychunk->dims[2], 256);
  if (chunk_hist == NULL) { ret = 1; goto cleanup; }
  hist_stats stats = vs_calculate_histogram_stats(slice_hist);
  printf("Mean: %.2f\n", stats.mean);
  printf("Median: %.2f\n", stats.median);
  printf("Mode: %.2f (count: %u)\n", stats.mode, stats.mode_count);
  printf("Standard Deviation: %.2f\n", stats.std_dev);

  if (vs_write_histogram_to_csv(slice_hist, "slice_histogram.csv")) { ret = 1; goto cleanup; }
  if (vs_write_histogram_to_csv(chunk_hist, "chunk_histogram.csv")) { ret = 1; goto cleanup; }

  cleanup:
  vs_histogram_free(slice_hist);
  vs_histogram_free(chunk_hist);
  vs_chunk_free(mychunk);
  vs_slice_free(myslice);
  vs_vol_free(vol);
  printf("%s done \n",__FUNCTION__);
  return ret;
}

int testzarr() {
  printf("%s\n",__FUNCTION__);
  int ret = 0;
  char* buf;
  long len = vs_download(TEST_ZARRAY_URL, &buf);

  zarr_metadata metadata;
  if (vs_zarr_parse_metadata(buf,&metadata)) {
    ret = 1;
    goto cleanup;
  }

  int z = metadata.chunks[0];
  int y = metadata.chunks[1];
  int x = metadata.chunks[2];
  if(!strcmp(metadata.dtype,"|u1") == 0) {
    ret = 1;
    goto cleanup;
  }

  u8* compressed_data;
  len = vs_download(TEST_ZARR_BLOCK_URL, &compressed_data);
  if (len <= 0) {
    ret = 1;
    goto cleanup;
  }

  unsigned char *decompressed_data = malloc(z * y * x);
  int decompressed_size = blosc2_decompress(compressed_data, len, decompressed_data, z*y*x);
  if (decompressed_size < 0) {
    fprintf(stderr, "Blosc2 decompression failed: %d\n", decompressed_size);
    ret = 1;
    goto cleanup;
  }
  cleanup:
  free(compressed_data);
  free(decompressed_data);
  free(buf);
  printf("%s done \n",__FUNCTION__);
  return ret;
}

int testmesher() {
  printf("%s\n", __FUNCTION__);
  int ret = 0;
  float* vertices = NULL;
  float* graycolors = NULL;
  int* indices = NULL;
  volume* vol = vs_vol_new(TEST_CACHEDIR, TEST_ZARR_URL);
  chunk* mychunk;
  if ((mychunk= vs_vol_get_chunk(vol, (s32[3]){2048,2048,2048},(s32[3]){256,128,128})) == NULL) {
    ret = 1; goto cleanup;
  }

  if (mychunk == NULL) { ret = 1; goto cleanup; }
  chunk* smallerchunk = vs_sumpool(mychunk, 2, 2);
  if (smallerchunk == NULL) { ret = 1; goto cleanup; }
  chunk* rescaled = vs_normalize_chunk(smallerchunk);
  if (rescaled == NULL) { ret = 1; goto cleanup; }

  int vertex_count, indices_count;
  ret = vs_march_cubes(rescaled->data, rescaled->dims[0], rescaled->dims[1], rescaled->dims[2], 0.5f, &vertices, &graycolors,
                        &indices, &vertex_count, &indices_count);
  if (ret != 0) { ret = 1; goto cleanup; }
  ret = vs_ply_write("mymesh.ply", vertices,NULL, NULL,indices, vertex_count, indices_count);
  if (ret != 0) { ret = 1; goto cleanup; }

  cleanup:
  free(indices);
  free(vertices);
  vs_chunk_free(rescaled);
  vs_chunk_free(mychunk);
  vs_chunk_free(smallerchunk);
  vs_vol_free(vol);
  printf("%s done \n",__FUNCTION__);
  return ret;
}

int testmath() {
  printf("%s\n", __FUNCTION__);
  int ret = 0;
  chunk* mychunk = vs_chunk_new((s32[3]){128, 128, 128});
  if (mychunk == NULL) { ret = 1; goto cleanup; }
  for (int z = 0; z < 128; z++) {
    for (int y = 0; y < 128; y++) {
      for (int x = 0; x < 128; x++) {
        vs_chunk_set(mychunk, z, y, x, 1.0f);
      }
    }
  }
  chunk* smaller = vs_sumpool(mychunk, 2, 2);
  if (smaller == NULL) { ret = 1; goto cleanup; }
  if (smaller->dims[0] != 64 || smaller->dims[1] != 64 || smaller->dims[2] != 64) { ret = 1; goto cleanup; }
  for (int z = 0; z < 64; z++) {
    for (int y = 0; y < 64; y++) {
      for (int x = 0; x < 64; x++) {
        f32 val = vs_chunk_get(smaller, z, y, x);
        if (val > 8.01f || val < 7.99f) { ret = 1; goto cleanup; }
      }
    }
  }

  cleanup:
  vs_chunk_free(smaller);
  vs_chunk_free(mychunk);
  printf("%s done \n",__FUNCTION__);
  return ret;
}

int testvcps() {

  printf("%s\n", __FUNCTION__);

  size_t width = 2, height = 2, dim = 3;
  size_t total_points = width * height * dim;

  // Test float->double->float conversion
  float* test_float_data = malloc(total_points * sizeof(float));
  for (size_t i = 0; i < total_points; i++) { test_float_data[i] = (float)i + 0.5f; }

  // Write float data as double
  if (vs_vcps_write("test_double.vcps", width, height, dim, test_float_data, "float", "double")) {
    fprintf(stderr, "Failed to write float->double test file\n");
    free(test_float_data);
    return 1;
  }

  // Read it back as float
  float* read_float_data = malloc(total_points * sizeof(float));
  size_t read_width, read_height, read_dim;

  int read_status = vs_vcps_read("test_double.vcps", &read_width, &read_height, &read_dim, read_float_data, "float");
  if (read_status) {
    fprintf(stderr, "Failed to read double->float test file (status=%d)\n", read_status);
    free(test_float_data);
    free(read_float_data);
    return 1;
  }

  // Verify dimensions
  if (width != read_width || height != read_height || dim != read_dim) {
    fprintf(stderr, "Dimension mismatch: expected (%zux%zux%zu), got (%zux%zux%zu)\n",
            width, height, dim, read_width, read_height, read_dim);
    free(test_float_data);
    free(read_float_data);
    return 1;
  }

  // Verify float data
  int float_test_passed = 1;
  for (size_t i = 0; i < total_points; i++) {
    if (fabsf(test_float_data[i] - read_float_data[i]) > 1e-6f) {
      fprintf(stderr, "Float data mismatch at %zu: %f != %f\n",
              i, test_float_data[i], read_float_data[i]);
      float_test_passed = 0;
      break;
    }
  }
  free(test_float_data);
  free(read_float_data);

  return float_test_passed ? 0 : 1;
}

int testchamfer() {
  int ret = 0;
  printf("%s\n", __FUNCTION__);
  float *vertices1 = NULL, *vertices2 = NULL;
  float *graycolors1 = NULL, *graycolors2 = NULL;
  int *indices1 = NULL, *indices2 = NULL;
  int vertex_count1, vertex_count2;
  int index_count1, index_count2;

  volume* vol = vs_vol_new(TEST_CACHEDIR, TEST_ZARR_URL);
  chunk* mychunk = vs_vol_get_chunk(vol, (s32[3]){2048,2048,2048},(s32[3]){128,128,128});

  ret = vs_march_cubes(mychunk->data, mychunk->dims[0], mychunk->dims[1], mychunk->dims[2], 128.0f, &vertices1, &graycolors1,
                          &indices1, &vertex_count1, &index_count1);
  if (ret != 0) { ret = 1; goto cleanup; }

  vs_chunk_free(mychunk);

  if ((mychunk = vs_vol_get_chunk(vol, (s32[3]){2048+64,2048+64,2048+64},(s32[3]){64,64,64}))== NULL) {
    ret = 1; goto cleanup;
  }
  ret = vs_march_cubes(mychunk->data, mychunk->dims[0], mychunk->dims[1], mychunk->dims[2], 128.0f, &vertices2, &graycolors2,
                            &indices2, &vertex_count2, &index_count2);
  if (ret != 0) { ret = 1; goto cleanup; }

  printf("calculating chamfer distance\n");
  float distance = vs_chamfer_distance(vertices1, vertex_count1, vertices2, vertex_count2);

  printf("Chamfer distance between meshes: %f\n", distance);

  cleanup:
  vs_chunk_free(mychunk);
  vs_vol_free(vol);
  free(vertices1);
  free(vertices2);
  free(indices1);
  free(indices2);
  printf("%s done \n",__FUNCTION__);
  return ret;
}

int testvol() {
  printf("%s\n", __FUNCTION__);
  int ret = 0;
  volume* vol = vs_vol_new(TEST_CACHEDIR, TEST_ZARR_URL);

  float *vertices = NULL;
  int *indices = NULL;
  int vertex_count;
  int index_count;
  float* graycolors = NULL;

  chunk* mychunk = NULL;


  for (int sz = 128; sz <= 256; sz+=128) {
    if ((mychunk = vs_vol_get_chunk(vol, (s32[3]){2048,2048,2048},(s32[3]){sz,sz,sz})) == NULL) {
      ret = 1; goto cleanup;
    }

    ret = vs_march_cubes(mychunk->data, mychunk->dims[0], mychunk->dims[1], mychunk->dims[2], 128.0f, &vertices, &graycolors,
                              &indices, &vertex_count, &index_count);

    if (ret != 0) {
      ret = 1; goto cleanup;
    }

    char out_filename[1024] = {'\0'};
    sprintf(out_filename,"mymesh%d.ply",sz);

    ret = vs_ply_write(out_filename, vertices,NULL, NULL, indices, vertex_count, index_count);
    if (ret != 0) {
      ret = 1; goto cleanup;
    }
    free(vertices);
    free(indices);
    vs_chunk_free(mychunk);
    mychunk = NULL;
  }

  cleanup:
  vs_chunk_free(mychunk);
  vs_vol_free(vol);
  printf("%s done \n",__FUNCTION__);
  return ret;
}

int main(int argc, char** argv) {
  if (testcurl())      printf("testcurl failed\n");
  if (testzarr())      printf("testzarr failed\n");
  if (testhistogram()) printf("testhistogram failed\n");
  if (testmesher())    printf("testmesher failed\n");
  if (testmath())      printf("testmath failed\n");
  if (testvcps())      printf("testvcps failed\n");
  if (testchamfer())   printf("testchamfer failed\n");
  if (testvol())       printf("testvol failed\n");


  return 0;
}
