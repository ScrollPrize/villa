### Segmentation growth and manipulation

VC3D offers a number of tools to facilitate the editing and growth of scroll segmentations. All growth is structured as optimization problems solved using the ceres solver, and most of the logic is found in the following files : 

- `core/src/GrowPatch.cpp` - the primary "patch" growth logic
- `core/src/GrowSurface.cpp` = the primary "trace" growth logic
- `core/include/vc/core/util/CostFunctions.hpp` - the cost functions used by the growth algorithms
- `core/include/vc/tracer/Tracer.hpp` - the actual function call used to initiate surface growth 

Note: While there exists a differentiation between "patches" and "traces" within the codebase, the actual resultant file is exactly the same between the two. The difference is simply in how they are grown. 

