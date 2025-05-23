# Partitioned Direct Accumulation Migration Plan

## Overview
Transition from the current 2-phase pipeline (inference → blending) to a single-phase partitioned direct accumulation system that eliminates intermediate patch storage and provides 50-70% speed improvement.

**Key Integration Point**: Work WITH `VCDataset`'s existing partition system, not replace it.

## Current vs New Architecture

### Current System
```
VCDataset (partitioned) → [GPU0: patches] → logits_part_0.zarr
                        → [GPU1: patches] → logits_part_1.zarr  
                        → [GPU2: patches] → logits_part_2.zarr
                        → [Blending Phase] → merged_logits.zarr
                        → [Finalization] → final_output.zarr
```

### New System
```
VCDataset (partitioned) → [GPU0: direct accumulator] → raw_accumulator_part_0.zarr
                        → [GPU1: direct accumulator] → raw_accumulator_part_1.zarr
                        → [GPU2: direct accumulator] → raw_accumulator_part_2.zarr
                        → [Boundary Merge + Finalize] → final_output.zarr
```

---

## Phase 1: VCDataset Integration Analysis

### 1.1 Understand VCDataset Partitioning
- [x] **Review how VCDataset handles partitioning**
  - Uses Z-axis partitioning: `z_start <= pos[0] < z_end`
  - Generates `all_positions` list with (z, y, x) coordinates
  - Already filters patches by `part_id` and `num_parts`
  - Supports empty patch skipping

- [x] **Analyze partition boundary overlaps**
  - [x] Document which patches near boundaries might be missed
  - [x] Calculate overlap requirements for Gaussian blending (50% overlap needed)
  - [x] Determine approach: overlapping partitions instead of boundary patch generation

### 1.2 Accumulator Integration Points
- [x] **Design accumulator that works with VCDataset coordinates**
  ```python
  # DESIGNED: Accumulator coordinate system based on implemented VCDataset changes
  class PartitionedAccumulator:
      def __init__(self, dataset, num_classes, gaussian_map, output_path):
          # Get partition bounds from modified VCDataset
          bounds = dataset.get_partition_bounds()
          volume_shape = dataset.input_shape[1:] if len(dataset.input_shape) == 4 else dataset.input_shape
          
          # Store partition information
          self.expanded_z_start = bounds['expanded_z_start']
          self.expanded_z_end = bounds['expanded_z_end']
          self.core_z_start = bounds['core_z_start']
          self.core_z_end = bounds['core_z_end']
          self.overlap_size = bounds['overlap_size']
          
          # Accumulator covers EXPANDED region (includes overlaps)
          expanded_z_size = self.expanded_z_end - self.expanded_z_start
          self.accumulator_shape = (num_classes, expanded_z_size, volume_shape[1], volume_shape[2])
          
          # Create zarr stores for logits and weights
          self.logits_store = open_zarr(f"{output_path}/logits", mode='w', 
                                      shape=self.accumulator_shape, dtype=np.float32)
          weights_shape = self.accumulator_shape[1:]  # No class dimension
          self.weights_store = open_zarr(f"{output_path}/weights", mode='w',
                                       shape=weights_shape, dtype=np.float32)
          
      def accumulate_patch(self, patch_logits, global_coords):
          # Convert global coordinates to local accumulator coordinates
          z_global, y_global, x_global = global_coords
          z_local = z_global - self.expanded_z_start
          
          # Accumulate with Gaussian weighting (implementation details in Phase 4)
          
      def get_core_region_data(self):
          # Extract only the CORE region data (non-overlapping) for final merging
          core_z_local_start = self.core_z_start - self.expanded_z_start
          core_z_local_end = self.core_z_end - self.expanded_z_start
          return self.logits_store[:, core_z_local_start:core_z_local_end, :, :]
  ```

---

## Phase 2: Modify VCDataset for Boundary Handling

### 2.1 ~~Add Boundary Patch Generation~~ → Implemented Overlapping Partitions
- [x] **Modified VCDataset to use overlapping partitions (50% overlap)**
  ```python
  # IMPLEMENTED: Modified Z-axis partitioning logic
  overlap_size = pZ // 2  # 50% of patch Z dimension
  z_start = max(0, core_z_start - overlap_size) if self.part_id > 0 else core_z_start
  z_end = min(max_z, core_z_end + overlap_size) if self.part_id < self.num_parts - 1 else core_z_end
  
  # Added partition bounds tracking
  self.core_z_start = core_z_start      # Non-overlapping region
  self.core_z_end = core_z_end
  self.expanded_z_start = z_start       # Expanded region with overlap
  self.expanded_z_end = z_end
  self.overlap_size = overlap_size
  ```

### 2.2 Coordinate Mapping for Partitions
- [x] **Add partition-aware coordinate methods to VCDataset**
  ```python
  # IMPLEMENTED: Added to VCDataset class
  def get_partition_bounds(self):
      """Return the Z-bounds for this partition (core and expanded regions)"""
      return {
          'core_z_start': self.core_z_start,
          'core_z_end': self.core_z_end,
          'expanded_z_start': self.expanded_z_start,
          'expanded_z_end': self.expanded_z_end,
          'overlap_size': self.overlap_size
      }
      
  def global_to_local_coords(self, global_coords):
      """Convert global volume coordinates to partition-local coordinates"""
      bounds = self.get_partition_bounds()
      z_global, y_global, x_global = global_coords
      z_local = z_global - bounds['expanded_z_start']
      return (z_local, y_global, x_global)
      
  def is_in_core_region(self, global_coords):
      """Check if coordinates are in this partition's core region"""
      bounds = self.get_partition_bounds()
      z_global, y_global, x_global = global_coords
      return bounds['core_z_start'] <= z_global < bounds['core_z_end']
  ```

---

## Phase 3: Modify inference.py for Direct Accumulation

### 3.1 Replace Patch Storage with Accumulation
- [x] **Remove patch zarr creation in `Inferer._create_output_stores()`**
  ```python
  # IMPLEMENTED: Removed entire _create_output_stores() method
  # No longer needed with direct accumulation approach
  ```

- [x] **Add accumulator creation method**
  ```python
  # IMPLEMENTED: Added _create_accumulator() method
  def _create_accumulator(self):
      """Create accumulator for direct accumulation instead of patch storage"""
      from models.run.partitioned_accumulator import PartitionedAccumulator
      
      # Create accumulator for this partition only
      self.accumulator = PartitionedAccumulator(
          dataset=self.dataset,
          num_classes=self.num_classes,
          gaussian_map=self.gaussian_map,
          output_path=os.path.join(self.output_dir, f"accumulator_part_{self.part_id}")
      )
  ```

### 3.2 Modify Batch Processing Loop
- [x] **Replace patch writing with accumulation in `_process_batches()`**
  ```python
  # IMPLEMENTED: Replaced entire method with direct accumulation
  # Removed threading, zarr writing, and complex futures handling
  # Added direct accumulation loop:
  for i in range(current_batch_size):
      patch_data = output_np[i]  # Shape: (C, Z, Y, X)
      patch_index = patch_indices[i] if i < len(patch_indices) else i
      
      # Get global coordinates for this patch
      global_coords = self.patch_start_coords_list[patch_index]
      
      # Accumulate directly
      self.accumulator.accumulate_patch(patch_data, global_coords)
  ```

### 3.3 Add Gaussian Map Generation
- [x] **Create Gaussian map during initialization**
  ```python
  # IMPLEMENTED: Added _create_gaussian_map() method
  def _create_gaussian_map(self):
      """Generate Gaussian weighting map for blending"""
      from models.run.blending import generate_gaussian_map
      self.gaussian_map = generate_gaussian_map(
          self.patch_size, 
          sigma_scale=8.0, 
          verbose=self.verbose
      )[0]  # Remove batch dimension
  ```

---

## Phase 4: Create PartitionedAccumulator Class

### 4.1 Core Accumulator Implementation
- [x] **Create new file: `models/run/partitioned_accumulator.py`**
  ```python
  # IMPLEMENTED: Full PartitionedAccumulator class with VCDataset integration
  class PartitionedAccumulator:
      def __init__(self, dataset, num_classes, gaussian_map, output_path):
          # Uses dataset.get_partition_bounds() for partition information
          # Gets expanded region bounds (core + overlaps) from VCDataset
          # Creates logits_store and weights_store as zarr arrays
          
      def accumulate_patch(self, patch_logits, global_coords):
          # Direct accumulation with Gaussian weighting
          # Converts global coords to local partition coordinates
          # Efficient numpy operations for weighted accumulation
          
      def get_core_region_data(self):
          # Extracts non-overlapping core region for final merging
          # Returns (core_logits, core_weights, core_bounds)
          
      def normalize_and_finalize(self, mode='binary', threshold=False, epsilon=1e-8):
          # In-place normalization with division by weights
          # Supports binary/multiclass/raw output modes
          # Integrated softmax/argmax processing
  ```

---

## Phase 5: Boundary Merging Strategy

### 5.1 Detect Partition Boundaries
- [x] **Create boundary detection logic**
  ```python
  # IMPLEMENTED: find_partition_boundaries() in boundary_merger.py
  def find_partition_boundaries(accumulator_paths, patch_size):
      """Find regions where partitions need to be merged based on core/expanded regions"""
      # - Extracts partition bounds from accumulator metadata
      # - Calculates overlap regions between adjacent partitions
      # - Returns boundary information for merging
  ```

### 5.2 Boundary Merging Process
- [x] **Create `BoundaryMerger` class**
  ```python
  # IMPLEMENTED: BoundaryMerger class in boundary_merger.py
  class BoundaryMerger:
      def merge_partitions(self, accumulator_paths, final_output_path, patch_size):
          """Merge overlapping regions between partitions"""
          # - Finds boundaries between partitions
          # - Copies core regions from each partition (no overlap)
          # - Merges boundary regions by combining weighted contributions
          # - Normalizes combined regions and writes to final output
          
      def _copy_core_region(self, accumulator_path, final_store, part_id):
          """Copy core (non-overlapping) region from partition to final output"""
          
      def _merge_boundary_region(self, boundary, final_store):
          """Merge overlapping boundary region between two partitions"""
          # - Loads overlapping data from both partitions
          # - Combines by summing weighted contributions
          # - Normalizes and writes to final output
  ```

---

## Phase 6: Pipeline Integration

### 6.1 Update vesuvius_pipeline.py
- [ ] **Add partitioned accumulation mode**
  ```python
  def run_partitioned_accumulation_pipeline(args):
      
      # Launch inference with direct accumulation per GPU
      accumulator_paths = []
      for part_id in range(num_partitions):
          gpu_id = part_id % len(args.gpus)
          accumulator_path = run_inference_with_accumulation(
              args, part_id, gpu_id, num_partitions
          )
          accumulator_paths.append(accumulator_path)
      
      # Merge boundaries and finalize
      final_output = merge_and_finalize(accumulator_paths, args.output, args.mode, args.threshold)
      
      return final_output
  ```


## Phase 7: Normalization & Finalization Integration

### 7.1 When to Normalize
**Strategy: Normalize during partition finalization, then merge**

- [ ] **Each partition normalizes independently**
  ```python
  def finalize_partition(self, mode='binary', threshold=False):
      # 1. Normalize: logits = accumulator / weights
      # 2. Apply softmax/argmax processing  
      # 3. Convert to final output format
      # 4. Write to partition's final output
  ```

### 7.2 Integrated Output Processing
- [ ] **Combine normalization + finalization**
  ```python
  def normalize_and_process_partition(accumulator, mode, threshold):
      # Read raw accumulated data
      raw_logits = accumulator.logits_store[:]
      weights = accumulator.weights_store[:]
      
      # Normalize
      normalized = raw_logits / (weights[np.newaxis, :, :, :] + 1e-8)
      
      # Convert to torch for processing
      logits_tensor = torch.from_numpy(normalized)
      
      # Apply finalization logic (from finalize_outputs.py)
      if mode == "binary":
          softmax = F.softmax(logits_tensor, dim=0)
          if threshold:
              output = (softmax[1] > softmax[0]).float().unsqueeze(0)
          else:
              output = softmax[1].unsqueeze(0)  # Foreground probability
      
      # Convert to uint8 and return
      return convert_to_final_format(output)
  ```

---

## Phase 8: Migration Strategy

### 8.2 Testing Strategy
- [ ] **Unit tests for new components**
  - [ ] Test `PartitionedAccumulator` with known data
  - [ ] Test boundary merging with synthetic overlaps
  - [ ] Validate memory usage estimates
  - [ ] Test VCDataset coordinate integration

- [ ] **Integration tests**
  - [ ] Multi-GPU coordination testing

---

## Implementation Checklist

### Week 1: Core Architecture
- [ ] Create `PartitionedAccumulator` class
- [ ] Add boundary detection logic
- [ ] Test accumulator with single partition

### Week 2: VCDataset Integration  
- [ ] Add partition bounds methods to VCDataset
- [ ] Modify inference.py to use accumulator
- [ ] Test single-GPU partitioned accumulation

### Week 3: Multi-GPU & Boundaries
- [ ] Implement boundary merging
- [ ] Add multi-GPU coordination
- [ ] Test cross-partition boundary handling

### Week 4: Pipeline Integration
- [ ] Update vesuvius_pipeline.py
- [ ] Add memory budget validation
- [ ] Performance testing and optimization

---

## Success Metrics

- [ ] **Performance**: 50-70% reduction in total processing time
- [ ] **Storage**: 50% reduction in peak storage usage
- [ ] **Quality**: Outputs identical to legacy pipeline (within numerical precision)
- [ ] **Memory**: Accumulator memory usage within GPU bounds
- [ ] **Reliability**: No race conditions or coordination failures