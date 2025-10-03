### Segmentation growth and manipulation

VC3D offers a number of tools to facilitate the editing and growth of scroll segmentations. All growth is structured as optimization problems solved using the ceres solver, and most of the logic is found in the following files : 

- `core/src/GrowPatch.cpp` - the primary "patch" growth logic
- `core/src/GrowSurface.cpp` = the primary "trace" growth logic
- `core/include/vc/core/util/CostFunctions.hpp` - the cost functions used by the growth algorithms
- `core/include/vc/tracer/Tracer.hpp` - the actual function call used to initiate surface growth 

> [!NOTE]
> While there exists a differentiation between "patches" and "traces" within the codebase and we have a tendency to refer to them as separate things, the actual resultant file is exactly the same between the two. The difference is simply in how they are grown. 

### Seeding patches / Initial patch creation 
For all growth or editing actions, you'll need an initial mesh to start from. This can be done either through the GUI or the CLI. 

*From the GUI* 
- Launch VC3D and open a .volpkg
- Select the seeding widget (on the right side dock)
- If not already selected, click `switch to draw mode` and draw a path _across_ a surface prediction (or multiple)
- Click `analyze paths` , and you should see a seed point (or multiple) on the surface prediction
- Ensure your seed.json exists in the .volpkg root 
- Click `run seeding` 
- Click `refresh surfaces` and you should see your segmentation in the surface list on the left side of the UI 

*From the terminal -- adjust for your own locations* 

<video width="320" height="240" controls>
  <source src="imgs/seeding_placement.mp4" type="video/mp4">
</video>

```bash
/home/sean/Documents/villa/volume-cartographer/build/bin/vc_grow_seg_from_seed \
  -v /mnt/raid_nvme/volpkgs/PHerc172.volpkg/volumes/s5_105.zarr \
  -t /mnt/raid_nvme/volpkgs/PHerc172.volpkg/paths \
  -p /mnt/raid_nvme/volpkgs/PHerc172.volpkg/seed.json \
  -s 1674.9 3066.41 6915.49
```