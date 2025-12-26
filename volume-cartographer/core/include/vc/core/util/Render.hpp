#pragma once


void render_binary_mask(QuadSurface* surf,
                         cv::Mat_<uint8_t>& mask,
                         cv::Mat_<cv::Vec3f>& coords_out,
                         float scale = 1.0f);

void render_surface_image(QuadSurface* surf,
                         cv::Mat_<uint8_t>& mask,
                         cv::Mat_<uint8_t>& img,
                         z5::Dataset* ds,
                         ChunkCache<uint8_t>* cache,
                         float scale = 1.0f);

void render_image_from_coords(const cv::Mat_<cv::Vec3f>& coords,
                              cv::Mat_<uint8_t>& img,
                              z5::Dataset* ds,
                              ChunkCache<uint8_t>* cache);