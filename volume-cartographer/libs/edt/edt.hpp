/* Multi-Label Anisotropic Euclidean Distance Transform 3D
 *
 * edt, edtsq - compute the euclidean distance transform 
 *     on a single or multi-labeled image all at once.
 *     boolean images are faster.
 *
 * binary_edt, binary_edtsq: Compute the EDT on a binary image
 *     for all input data types. Multiple labels are not handled
 *     but it's faster.
 *
 * Author: William Silversmith
 * Affiliation: Seung Lab, Princeton Neuroscience Insitute
 * Date: July 2018
 */

#ifndef EDT_H
#define EDT_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include "threadpool.h"

// The pyedt namespace contains the primary implementation,
// but users will probably want to use the edt namespace (bottom)
// as the function sigs are a bit cleaner.
// pyedt names are underscored to prevent namespace collisions
// in the Cython wrapper.

namespace pyedt {

template <typename T>
inline double sq(T x) {
  return static_cast<double>(x) * static_cast<double>(x);
}

inline void tofinite(float *f, const int64_t voxels) {
  for (int64_t i = 0; i < voxels; i++) {
    if (std::isinf(f[i])) {
      f[i] = std::numeric_limits<float>::max() - 1;
    }
  }
}

inline void toinfinite(float *f, const int64_t voxels) {
  for (int64_t i = 0; i < voxels; i++) {
    if (f[i] >= std::numeric_limits<float>::max() - 1) {
      f[i] = INFINITY;
    }
  }
}

/* 1D Euclidean Distance Transform for Multiple Segids
 *
 * Map a row of segids to a euclidean distance transform.
 * Zero is considered a universal boundary as are differing
 * segids. Segments touching the boundary are mapped to 1.
 *
 * T* segids: 1d array of (un)signed integers
 * *d: write destination, equal sized array as *segids
 * n: size of segids, d
 * stride: typically 1, but can be used on a 
 *    multi dimensional array, in which case it is nx, nx*ny, etc
 * anisotropy: physical distance of each voxel
 *
 * Writes output to *d
 */
template <typename T>
inline void squared_edt_1d_multi_seg(
    T* segids, float *d, const int64_t n, 
    const int64_t stride, const float anistropy,
    const bool black_border=false
  ) {

  if (n == 0) {
    return;
  }

  int64_t i;

  T working_segid = segids[0];

  if (black_border) {
    d[0] = static_cast<float>(working_segid != 0) * anistropy; // 0 or 1
  }
  else {
    d[0] = working_segid == 0 ? 0 : INFINITY;
  }

  for (i = stride; i < n * stride; i += stride) {
    if (segids[i] == 0) {
      d[i] = 0.0;
    }
    else if (segids[i] == working_segid) {
      d[i] = d[i - stride] + anistropy;
    }
    else {
      d[i] = anistropy;
      d[i - stride] = static_cast<float>(segids[i - stride] != 0) * anistropy;
      working_segid = segids[i];
    }
  }

  int64_t min_bound = 0;
  if (black_border) {
    d[n - stride] = static_cast<float>(segids[n - stride] != 0) * anistropy;
    min_bound = stride;
  }

  for (i = (n - 2) * stride; i >= min_bound; i -= stride) {
    d[i] = std::fminf(d[i], d[i + stride] + anistropy);
  }

  for (i = 0; i < n * stride; i += stride) {
    d[i] *= d[i];
  }
}

/* 1D Euclidean Distance Transform based on:
 * 
 * http://cs.brown.edu/people/pfelzens/dt/
 * 
 * Felzenszwalb and Huttenlocher. 
 * Distance Transforms of Sampled Functions.
 * Theory of Computing, Volume 8. p415-428. 
 * (Sept. 2012) doi: 10.4086/toc.2012.v008a019
 *
 * Essentially, the distance function can be 
 * modeled as the lower envelope of parabolas
 * that spring mainly from edges of the shape
 * you want to transform. The array is scanned
 * to find the parabolas, then a second scan
 * writes the correct values.
 *
 * O(N) time complexity.
 *
 * I (wms) make a few modifications for our use case
 * of executing a euclidean distance transform on
 * a 3D anisotropic image that contains many segments
 * (many binary images). This way we do it correctly
 * without running EDT > 100x in a 512^3 chunk.
 *
 * The first modification is to apply an envelope 
 * over the entire volume by defining two additional
 * vertices just off the ends at x=-1 and x=n. This
 * avoids needing to create a black border around the
 * volume (and saves 6s^2 additional memory).
 *
 * The second, which at first appeared to be important for
 * optimization, but after reusing memory appeared less important,
 * is to avoid the division operation in computing the intersection
 * point. I describe this manipulation in the code below.
 *
 * I make a third modification in squared_edt_1d_parabolic_multi_seg
 * to enable multiple segments.
 *
 * Parameters:
 *   *f: the image ("sampled function" in the paper)
 *    *d: write destination, same size in voxels as *f
 *    n: number of voxels in *f
 *    stride: 1, sx, or sx*sy to handle multidimensional arrays
 *    anisotropy: e.g. (4nm, 4nm, 40nm)
 * 
 * Returns: writes distance transform of f to d
 */
inline void squared_edt_1d_parabolic(
    float* f,
    const int64_t n, 
    const int64_t stride, 
    const float anisotropy, 
    const bool black_border_left,
    const bool black_border_right
  ) {

  if (n == 0) {
    return;
  }

  const double w2 = anisotropy * anisotropy;

  int64_t k = 0;
  std::unique_ptr<int64_t[]> v(new int64_t[n]);
  v[0] = 0;

  std::unique_ptr<double[]> ff(new double[n]);
  for (int64_t i = 0; i < n; i++) {
    ff[i] = f[i * stride];
  }
  
  std::unique_ptr<double[]> ranges(new double[n + 1]);

  ranges[0] = -INFINITY;
  ranges[1] = +INFINITY;

  /* Unclear if this adds much but I certainly find it easier to get the parens right.
   *
   * Eqn: s = ( f(r) + r^2 ) - ( f(p) + p^2 ) / ( 2r - 2p )
   * 1: s = (f(r) - f(p) + (r^2 - p^2)) / 2(r-p)
   * 2: s = (f(r) - r(p) + (r+p)(r-p)) / 2(r-p) <-- can reuse r-p, replace mult w/ add
   */
  double s;
  double factor1, factor2;
  for (int64_t i = 1; i < n; i++) {
    factor1 = static_cast<double>(i - v[k]) * w2;
    factor2 =  static_cast<double>(i + v[k]);
    s = (ff[i] - ff[v[k]] + factor1 * factor2) / (2.0 * factor1);

    while (k > 0 && s <= ranges[k]) {
      k--;
      factor1 = static_cast<double>(i - v[k]) * w2;
      factor2 =  static_cast<double>(i + v[k]);
      s = (ff[i] - ff[v[k]] + factor1 * factor2) / (2.0 * factor1);
    }

    k++;
    v[k] = i;
    ranges[k] = s;
    ranges[k + 1] = +INFINITY;
  }

  k = 0;
  double envelope;
  for (int64_t i = 0; i < n; i++) {
    while (ranges[k + 1] < i) { 
      k++;
    }

    f[i * stride] = static_cast<float>(w2 * sq(i - v[k]) + ff[v[k]]);
    // Two lines below only about 3% of perf cost, thought it would be more
    // They are unnecessary if you add a black border around the image.
    if (black_border_left && black_border_right) {
      envelope = std::fmin(w2 * sq(i + 1), w2 * sq(n - i));
      f[i * stride] = std::fminf(static_cast<float>(envelope), f[i * stride]);
    }
    else if (black_border_left) {
      f[i * stride] = std::fminf(w2 * sq(i + 1), static_cast<float>(f[i * stride]));
    }
    else if (black_border_right) {
      f[i * stride] = std::fminf(w2 * sq(n - i), static_cast<float>(f[i * stride]));
    }
  }
}

// about 5% faster
inline void squared_edt_1d_parabolic(
    float* f,
    const int64_t n, 
    const int64_t stride, 
    const float anisotropy
  ) {

  if (n == 0) {
    return;
  }

  const double w2 = anisotropy * anisotropy;

  int64_t k = 0;
  std::unique_ptr<int64_t[]> v(new int64_t[n]);
  v[0] = 0;
  
  std::unique_ptr<double[]> ff(new double[n]);
  for (int64_t i = 0; i < n; i++) {
    ff[i] = f[i * stride];
  }

  std::unique_ptr<double[]> ranges(new double[n + 1]);

  ranges[0] = -INFINITY;
  ranges[1] = +INFINITY;

  /* Unclear if this adds much but I certainly find it easier to get the parens right.
   *
   * Eqn: s = ( f(r) + r^2 ) - ( f(p) + p^2 ) / ( 2r - 2p )
   * 1: s = (f(r) - f(p) + (r^2 - p^2)) / 2(r-p)
   * 2: s = (f(r) - r(p) + (r+p)(r-p)) / 2(r-p) <-- can reuse r-p, replace mult w/ add
   */
  double s;
  double factor1, factor2;
  for (int64_t i = 1; i < n; i++) {
    factor1 = static_cast<double>(i - v[k]) * w2;
    factor2 = static_cast<double>(i + v[k]);
    s = (ff[i] - ff[v[k]] + factor1 * factor2) / (2.0 * factor1);

    while (k > 0 && s <= ranges[k]) {
      k--;
      factor1 = static_cast<double>(i - v[k]) * w2;
      factor2 = static_cast<double>(i + v[k]);
      s = (ff[i] - ff[v[k]] + factor1 * factor2) / (2.0 * factor1);
    }

    k++;
    v[k] = i;
    ranges[k] = s;
    ranges[k + 1] = +INFINITY;
  }

  k = 0;
  double envelope;
  for (int64_t i = 0; i < n; i++) {
    while (ranges[k + 1] < i) { 
      k++;
    }

    f[i * stride] = w2 * sq(i - v[k]) + ff[v[k]];
    // Two lines below only about 3% of perf cost, thought it would be more
    // They are unnecessary if you add a black border around the image.
    envelope = std::fmin(w2 * sq(i + 1), w2 * sq(n - i));
    f[i * stride] = std::fminf(static_cast<float>(envelope), f[i * stride]);
  }
}

inline void _squared_edt_1d_parabolic(
    float* f, 
    const int64_t n, 
    const int64_t stride, 
    const float anisotropy, 
    const bool black_border_left,
    const bool black_border_right
  ) {

  if (black_border_left && black_border_right) {
    squared_edt_1d_parabolic(f, n, stride, anisotropy);
  }
  else {
    squared_edt_1d_parabolic(f, n, stride, anisotropy, black_border_left, black_border_right); 
  }
}

/* Same as squared_edt_1d_parabolic except that it handles
 * a simultaneous transform of multiple labels (like squared_edt_1d_multi_seg).
 * 
 *  Parameters:
 *    *segids: an integer labeled image where 0 is background
 *    *f: the image ("sampled function" in the paper)
 *    n: number of voxels in *f
 *    stride: 1, sx, or sx*sy to handle multidimensional arrays
 *    anisotropy: e.g. (4.0 = 4nm, 40.0 = 40nm)
 * 
 * Returns: writes squared distance transform in f
 */
template <typename T>
inline void squared_edt_1d_parabolic_multi_seg(
  T* segids, float* f,
  const int64_t n, const int64_t stride, const float anisotropy,
  const bool black_border=false
) {

  T working_segid = segids[0];
  T segid;
  int64_t last = 0;

  for (int64_t i = 1; i < n; i++) {
    segid = segids[i * stride];
    if (segid != working_segid) {
      if (working_segid != 0) {
        _squared_edt_1d_parabolic(
          f + last * stride, 
          i - last, stride, anisotropy,
          (black_border || last > 0), true
        );
      }
      working_segid = segid;
      last = i;
    }
  }

  if (working_segid != 0 && last < n) {
    _squared_edt_1d_parabolic(
      f + last * stride, 
      n - last, stride, anisotropy,
      (black_border || last > 0), black_border
    );
  }
}

/* Df(x,y,z) = min( wx^2 * (x-x')^2 + Df|x'(y,z) )
 *              x'                   
 * Df(y,z) = min( wy^2 * (y-y') + Df|x'y'(z) )
 *            y'
 * Df(z) = wz^2 * min( (z-z') + i(z) )
 *          z'
 * i(z) = 0   if voxel in set (f[p] == 1)
 *        inf if voxel out of set (f[p] == 0)
 *
 * In english: a 3D EDT can be accomplished by
 *    taking the x axis EDT, followed by y, followed by z.
 * 
 * The 2012 paper by Felzenszwalb and Huttenlocher describes using
 * an indicator function (above) to use their sampled function
 * concept on all three axes. This is unnecessary. The first
 * transform (x here) can be done very dumbly and cheaply using
 * the method of Rosenfeld and Pfaltz (1966) in 1D (where the L1
 * and L2 norms agree). This first pass is extremely fast and so
 * saves us about 30% in CPU time. 
 *
 * The second and third passes use the Felzenszalb and Huttenlocher's
 * method. The method uses a scan then write sequence, so we are able
 * to write to our input block, which increases cache coherency and
 * reduces memory usage.
 *
 * Parameters:
 *    *labels: an integer labeled image where 0 is background
 *    sx, sy, sz: size of the volume in voxels
 *    wx, wy, wz: physical dimensions of voxels (weights)
 *
 * Returns: writes squared distance transform of f to d
 */
template <typename T>
inline float* _edt3dsq(
  T* labels, 
  const int64_t sx, const int64_t sy, const int64_t sz, 
  const float wx, const float wy, const float wz,
  const bool black_border = false,
  const int parallel = 1,
  float* workspace = NULL
) {

  const int64_t sxy = sx * sy;
  const int64_t voxels = sz * sxy;

  if (workspace == NULL) {
    workspace = new float[sx * sy * sz]();
  }

  ThreadPool pool(parallel);

  for (int64_t z = 0; z < sz; z++) {
    pool.enqueue([labels, sy, z, sx, sxy, wx, workspace, black_border](){
      for (int64_t y = 0; y < sy; y++) {
        squared_edt_1d_multi_seg<T>(
          (labels + sx * y + sxy * z), 
          (workspace + sx * y + sxy * z), 
          sx, 1, wx, black_border
        ); 
      }
    });
  }

  pool.join();

  if (!black_border) {
    tofinite(workspace, voxels);
  }

  pool.start(parallel);

  for (int64_t z = 0; z < sz; z++) {
    pool.enqueue([labels, sxy, z, workspace, sx, sy, wy, black_border](){
      for (int64_t x = 0; x < sx; x++) {
        squared_edt_1d_parabolic_multi_seg<T>(
          (labels + x + sxy * z),
          (workspace + x + sxy * z), 
          sy, sx, wy, black_border
        );
      }
    });
  }

  pool.join();
  pool.start(parallel);

  for (int64_t y = 0; y < sy; y++) {
    pool.enqueue([labels, sx, y, workspace, sz, sxy, wz, black_border](){
      for (int64_t x = 0; x < sx; x++) {
        squared_edt_1d_parabolic_multi_seg<T>(
          (labels + x + sx * y), 
          (workspace + x + sx * y), 
          sz, sxy, wz, black_border
        );
      }
    });
  }

  pool.join();

  if (!black_border) {
    toinfinite(workspace, voxels);
  }

  return workspace; 
}

// skipping multi-seg logic results in a large speedup
template <typename T>
inline float* _binary_edt3dsq(
  T* binaryimg, 
  const int64_t sx, const int64_t sy, const int64_t sz, 
  const float wx, const float wy, const float wz,
  const bool black_border=false, const int parallel=1, 
  float* workspace=NULL
) {

  const int64_t sxy = sx * sy;
  const int64_t voxels = sz * sxy;

  int64_t x,y,z;

  if (workspace == NULL) {
    workspace = new float[sx * sy * sz]();
  }  

  ThreadPool pool(parallel);
  
  for (z = 0; z < sz; z++) {
    for (y = 0; y < sy; y++) { 
      pool.enqueue([binaryimg, sx, y, sxy, z, workspace, wx, black_border](){
        squared_edt_1d_multi_seg<T>(
          (binaryimg + sx * y + sxy * z), 
          (workspace + sx * y + sxy * z), 
          sx, 1, wx, black_border
        ); 
      });
    }
  }

  pool.join();

  if (!black_border) {
    tofinite(workspace, voxels);
  }

  pool.start(parallel);

  int64_t offset;
  for (z = 0; z < sz; z++) {
    for (x = 0; x < sx; x++) {
      offset = x + sxy * z;
      for (y = 0; y < sy; y++) {
        if (workspace[offset + sx*y]) {
          break;
        }
      }

      pool.enqueue([sx, sy, y, workspace, wy, black_border, offset](){
        _squared_edt_1d_parabolic(
          (workspace + offset + sx * y),
          sy - y, sx, wy, 
          black_border || (y > 0), black_border
        );
      });
    }
  }

  pool.join();
  pool.start(parallel);

  for (y = 0; y < sy; y++) {
    for (x = 0; x < sx; x++) {
      offset = x + sx * y;
      pool.enqueue([sz, sxy, workspace, wz, black_border, offset](){
        int64_t z = 0;
        for (z = 0; z < sz; z++) {
          if (workspace[offset + sxy*z]) {
            break;
          }
        }
        _squared_edt_1d_parabolic(
          (workspace + offset + sxy * z), 
          sz - z, sxy, wz, 
          black_border || (z > 0), black_border
        );
      });
    }
  }

  pool.join();

  if (!black_border) {
    toinfinite(workspace, voxels);
  }

  return workspace; 
}

// about 20% faster on binary images by skipping
// multisegment logic in parabolic
template <typename T>
inline float* _edt3dsq(bool* binaryimg, 
  const int64_t sx, const int64_t sy, const int64_t sz, 
  const float wx, const float wy, const float wz, 
  const bool black_border=false, const int parallel=1, float* workspace=NULL) {

  return _binary_edt3dsq(binaryimg, sx, sy, sz, wx, wy, wz, black_border, parallel, workspace);
}

// Same as _edt3dsq, but applies square root to get
// euclidean distance.
template <typename T>
inline float* _edt3d(T* input, 
  const int64_t sx, const int64_t sy, const int64_t sz, 
  const float wx, const float wy, const float wz,
  const bool black_border=false, const int parallel=1, float* workspace=NULL) {

  float* transform = _edt3dsq<T>(input, sx, sy, sz, wx, wy, wz, black_border, parallel, workspace);

  for (int64_t i = 0; i < sx * sy * sz; i++) {
    transform[i] = std::sqrt(transform[i]);
  }

  return transform;
}

// skipping multi-seg logic results in a large speedup
template <typename T>
inline float* _binary_edt3d(
    T* input, 
    const int64_t sx, const int64_t sy, const int64_t sz, 
    const float wx, const float wy, const float wz,
    const bool black_border=false, const int parallel=1, 
    float* workspace=NULL
  ) {

  float* transform = _binary_edt3dsq<T>(
    input, 
    sx, sy, sz, 
    wx, wy, wz, 
    black_border, parallel, 
    workspace
  );

  for (int64_t i = 0; i < sx * sy * sz; i++) {
    transform[i] = std::sqrt(transform[i]);
  }

  return transform;
}

// 2D version of _edt3dsq
template <typename T>
inline float* _edt2dsq(
    T* input, 
    const int64_t sx, const int64_t sy,
    const float wx, const float wy,
    const bool black_border=false, const int parallel=1,
    float* workspace=NULL
  ) {

  const int64_t voxels = sx * sy;

  if (workspace == NULL) {
    workspace = new float[voxels]();
  }

  for (int64_t y = 0; y < sy; y++) { 
    squared_edt_1d_multi_seg<T>(
      (input + sx * y), (workspace + sx * y), 
      sx, 1, wx, black_border
    ); 
  }

  if (!black_border) {
    tofinite(workspace, voxels);
  }

  ThreadPool pool(parallel);

  for (int64_t x = 0; x < sx; x++) {
    pool.enqueue([input, x, workspace, sy, sx, wy, black_border](){
      squared_edt_1d_parabolic_multi_seg<T>(
        (input + x), 
        (workspace + x), 
        sy, sx, wy,
        black_border
      );
    });
  }

  pool.join();

  if (!black_border) {
    toinfinite(workspace, voxels);
  }

  return workspace;
}

// skipping multi-seg logic results in a large speedup
template <typename T>
inline float* _binary_edt2dsq(T* binaryimg, 
  const int64_t sx, const int64_t sy,
  const float wx, const float wy,
  const bool black_border=false, const int parallel=1,
  float* workspace=NULL) {

  const int64_t voxels = sx * sy;
  int64_t x,y;

  if (workspace == NULL) {
    workspace = new float[sx * sy]();
  }

  for (y = 0; y < sy; y++) { 
    squared_edt_1d_multi_seg<T>(
      (binaryimg + sx * y), (workspace + sx * y), 
      sx, 1, wx, black_border
    ); 
  }

  if (!black_border) {
    tofinite(workspace, voxels);
  }

  ThreadPool pool(parallel);

  for (x = 0; x < sx; x++) {
    pool.enqueue([workspace, x, sx, sy, wy, black_border](){
      int64_t y = 0;
      for (y = 0; y < sy; y++) {
        if (workspace[x + y * sx]) {
          break;
        }
      }

      _squared_edt_1d_parabolic(
        (workspace + x + y * sx), 
        sy - y, sx, wy,
        black_border || (y > 0), black_border
      );
    });
  }

  pool.join();

  if (!black_border) {
    toinfinite(workspace, voxels);
  }

  return workspace;
}

// skipping multi-seg logic results in a large speedup
template <typename T>
inline float* _binary_edt2d(T* binaryimg, 
  const int64_t sx, const int64_t sy,
  const float wx, const float wy,
  const bool black_border=false, const int parallel=1,
  float* output=NULL) {

  float *transform = _binary_edt2dsq(
    binaryimg, 
    sx, sy, 
    wx, wy, 
    black_border, parallel, 
    output
  );

  for (int64_t i = 0; i < sx * sy; i++) {
    transform[i] = std::sqrt(transform[i]);
  }

  return transform;
}

// 2D version of _edt3dsq
template <typename T>
inline float* _edt2dsq(bool* binaryimg, 
  const int64_t sx, const int64_t sy,
  const float wx, const float wy,
  const bool black_border=false, const int parallel=1,
  float* output=NULL) {

  return _binary_edt2dsq(
    binaryimg, 
    sx, sy, 
    wx, wy, 
    black_border, parallel,
    output
  );
}

// returns euclidean distance instead of squared distance
template <typename T>
inline float* _edt2d(
    T* input, 
    const int64_t sx, const int64_t sy,
    const float wx, const float wy,
    const bool black_border=false, const int parallel=1,
    float* output=NULL
  ) {

  float* transform = _edt2dsq<T>(
    input, 
    sx, sy, 
    wx, wy, 
    black_border, parallel, 
    output
  );

  for (int64_t i = 0; i < sx * sy; i++) {
    transform[i] = std::sqrt(transform[i]);
  }

  return transform;
}


// Should be trivial to make an N-d version
// if someone asks for it. Might simplify the interface.

} // namespace pyedt

namespace edt {

template <typename T>
inline float* edt(
  T* labels, 
  const int sx, const float wx, 
  const bool black_border=false) {

  float* d = new float[sx]();
  pyedt::squared_edt_1d_multi_seg(labels, d, sx, 1, wx);

  for (int i = 0; i < sx; i++) {
    d[i] = std::sqrt(d[i]);
  }

  return d;
}

template <typename T>
inline float* edt(
    T* labels, 
    const int sx, const int sy, 
    const float wx, const float wy,
    const bool black_border=false, const int parallel=1,
    float* output=NULL
  ) {

  return pyedt::_edt2d(labels, sx, sy, wx, wy, black_border, parallel, output);
}


template <typename T>
inline float* edt(
  T* labels, 
  const int sx, const int sy, const int sz, 
  const float wx, const float wy, const float wz,
  const bool black_border=false, const int parallel=1, float* output=NULL) {

  return pyedt::_edt3d(labels, sx, sy, sz, wx, wy, wz, black_border, parallel, output);
}

template <typename T>
inline float* binary_edt(
  T* labels, 
  const int sx, 
  const float wx, 
  const bool black_border=false) {

  return edt::edt(labels, sx, wx, black_border);
}

template <typename T>
inline float* binary_edt(
    T* labels, 
    const int sx, const int sy, 
    const float wx, const float wy, 
    const bool black_border=false, const int parallel=1,
    float* output=NULL
  ) {

  return pyedt::_binary_edt2d(
    labels, 
    sx, sy, 
    wx, wy, 
    black_border, parallel, 
    output
  );
}

template <typename T>
inline float* binary_edt(
  T* labels, 
  const int sx, const int sy, const int sz, 
  const float wx, const float wy, const float wz,
  const bool black_border=false, const int parallel=1, float* output=NULL) {

  return pyedt::_binary_edt3d(labels, sx, sy, sz, wx, wy, wz, black_border, parallel, output);
}

template <typename T>
inline float* edtsq(
  T* labels, 
  const int sx, const float wx, 
  const bool black_border=false) {

  float* d = new float[sx]();
  pyedt::squared_edt_1d_multi_seg(labels, d, sx, 1, wx, black_border);
  return d;
}

template <typename T>
inline float* edtsq(
    T* labels, 
    const int sx, const int sy, 
    const float wx, const float wy,
    const bool black_border=false, const int parallel=1,
    float* output=NULL
  ) {

  return pyedt::_edt2dsq(labels, sx, sy, wx, wy, black_border, parallel, output);
}

template <typename T>
inline float* edtsq(
    T* labels, 
    const int sx, const int sy, const int sz, 
    const float wx, const float wy, const float wz,
    const bool black_border=false, const int parallel=1, 
    float* output=NULL
  ) {

  return pyedt::_edt3dsq(
    labels, 
    sx, sy, sz, 
    wx, wy, wz, 
    black_border, parallel, output
  );
}

template <typename T>
inline float* binary_edtsq(
  T* labels, 
  const int sx, const float wx, 
  const bool black_border=false, const int parallel=1) {

  return edt::edtsq(labels, sx, wx, black_border);
}

template <typename T>
inline float* binary_edtsq(
  T* labels, 
  const int sx, const int sy, 
  const float wx, const float wy,
  const bool black_border=false, const int parallel=1) {

  return pyedt::_binary_edt2dsq(labels, sx, sy, wx, wy, black_border, parallel);
}

template <typename T>
inline float* binary_edtsq(
  T* labels, 
  const int sx, const int sy, const int sz, 
  const float wx, const float wy, const float wz,
  const bool black_border=false, const int parallel=1, float* output=NULL) {

  return pyedt::_binary_edt3dsq(labels, sx, sy, sz, wx, wy, wz, parallel, output);
}


} // namespace edt

#undef sq

#endif

