#pragma once
#include <cstddef>
#include <cstdint>
namespace vcr::sampling {
// Float32 channel-first arrays: B,C,[Z,]Y,X; points B,Q,D in array axis order.
// Resident arrays never perform I/O. device=-1 selects CPU; nonnegative selects CUDA.
enum class Boundary { Zeros, Border, Constant };
struct Dense {
    const float* values=nullptr;
    const float* points=nullptr;
    std::size_t batches=0, channels=0, queries=0;
    int dimensions=3;
    std::int64_t shape[3]{1,1,1};
    int order=1;
    Boundary boundary=Boundary::Zeros;
    float fill=0;
};
// Outputs B,C,Q. Gradients are a vector-Jacobian product, in the input layouts.
void sampleDense(const Dense&, float* output, int device=-1);
void backwardDense(const Dense&, const float* cotangent, float* valuesGradient, float* pointsGradient, int device=-1);
// Surface batches B,H,W,3 and masks B,H,W. Queries B,Q,2 are row,column.
// Prepared derivatives are B,3,H,W,3 (column, row, mixed XYZ planes).
void prepareSurfaces(const float* xyz,const bool* valid,int batches,int rows,int cols,float* derivatives,int device=-1);
void sampleSurfaces(const float* xyz,const bool* valid,const float* derivatives,int batches,int rows,int cols,
                    const float* rc,std::size_t queries,float* positions,float* rowTangents,float* columnTangents,
                    bool* supported,int mode=2,bool border=false,int device=-1);

struct Tile {
    int rows, cols, outputRows, outputCols;
    int rowStart, rowEnd, colStart, colEnd;
};
// Pixel-centre output coordinates, replicated grid borders, scalar output.
void sampleFieldTile(const float* field,const bool* valid,const Tile&,float* output,bool* supported,int device=-1);
void sampleSurfaceTile(const float* xyz,const bool* valid,const float* derivatives,const Tile&,int mode,float* output,bool* supported,int device=-1);
}
