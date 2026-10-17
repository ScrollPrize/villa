#include "local_field.cuh"
namespace vcr::augment::cuda {
__global__ void expand_local(double* field, Layout g, const double* coefficients, double low, double range) {
    EACH_INDEX(g.size) {
        double value = 1; std::size_t offset = 0;
        for (int a = 0; a < g.dim; ++a) {
            value *= coefficients[offset+g.coord(i,a)]; offset += g.shape[a];
        }
        field[i] = range ? (value-low)/range : 0;
    }
}
__global__ void center_local(double* field, std::size_t n, const Statistics* stats) {
    const double m = stats->mean, maxabs = fmax(fabs(stats->low-m),fabs(stats->high-m));
    EACH_INDEX(n) { double value = field[i]-m; field[i] = maxabs > 1e-8 ? value/maxabs : value; }
}
std::shared_ptr<Buffer<double>> local_field(const LocalOptions& options, const Grid& grid, Random& rng, bool centered) {
    auto axes = local_axes(options,grid,rng);
    std::vector<double> coefficients; double low = 1, high = 1;
    for (const auto& axis : axes) {
        auto [lo,hi] = std::minmax_element(axis.begin(),axis.end()); low *= *lo; high *= *hi;
        coefficients.insert(coefficients.end(),axis.begin(),axis.end());
    }
    Buffer<double> table(coefficients); auto result = std::make_shared<Buffer<double>>(grid.size);
    expand_local<<<blocks(grid.size),256,0,vcr::device::stream()>>>(result->ptr,Layout(grid),table.ptr,low,high-low);
    if (centered) {
        auto stats = statistics(View<double>{result->ptr,result->size});
        center_local<<<blocks(grid.size),256,0,vcr::device::stream()>>>(result->ptr,result->size,stats.ptr);
    }
    check(cudaGetLastError()); return result;
}
} // namespace vcr::augment::cuda
