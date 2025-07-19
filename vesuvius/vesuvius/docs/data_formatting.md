# Data Structuring Guide for Vesuvius Python Training

This guide explains how to structure your data and configure labels for different segmentation scenarios using Vesuvius Python: binary segmentation, multi-class segmentation, and multi-task learning.

## Table of Contents
1. [Binary Segmentation](#binary-segmentation)
2. [Multi-class Segmentation](#multi-class-segmentation)
3. [Multi-task Learning](#multi-task-learning)
4. [Label Value Mapping](#label-value-mapping)
5. [Validation](#validation)

---

## Binary Segmentation

Binary segmentation involves classifying each pixel/voxel as either background (0) or foreground (non-zero).

### Napari Dataset Structure

For napari, organize your layers as follows:
```
image_layer_name: "my_image"
label_layer_name: "my_image_ink"  # Format: {image_name}_{task_name}
mask_layer_name: "my_image_mask"  # Optional
```

**Example:**
- Image layer: `"scroll_001"`
- Label layer: `"scroll_001_ink"`
- Mask layer: `"scroll_001_mask"` (optional)

### TIF/Zarr Dataset Structure

Directory structure:
```
data_path/
├── images/
│   └── image1_ink.tif
├── labels/
│   └── image1_ink.tif
└── masks/          # Optional
    └── image1_ink.tif
```

**File naming convention:** `{image_id}_{task_name}.{extension}`

### Configuration

```yaml
dataset_config:
  binarize_labels: true
  target_value:
    ink: 1  # Any non-zero pixels in the label will become 1

# Note: targets configuration is set automatically by the system
# based on your dataset_config. The system will configure:
# - channels: 1 (for binary segmentation)
# - activation: "sigmoid"
# - loss_fn: Based on configuration or defaults
```

### Label Values
- Background: 0
- Foreground: Any non-zero value (will be mapped to 1)

---

## Multi-class Segmentation

Multi-class segmentation assigns each pixel/voxel to one of several classes (mutually exclusive).

### Napari Dataset Structure

```
image_layer_name: "my_image"
label_layer_name: "my_image_segmentation"  # Single layer with multiple classes
mask_layer_name: "my_image_mask"          # Optional
```

**Example:**
- Image layer: `"fragment_1"`
- Label layer: `"fragment_1_ink"` (contains values 0, 1, 2, 3, etc.)
- Mask layer: `"fragment_1_mask"` (optional)

### TIF/Zarr Dataset Structure

```
data_path/
├── images/
│   └── image1.tif
├── labels/
│   └── image1_labels.tif  # Contains multi-class labels
└── masks/
    └── image1_mask.tif
```

### Configuration

#### Simple Multi-class (No Remapping)
```yaml
dataset_config:
  binarize_labels: false  # Use labels as-is
  target_value: null

# OR with identity mapping
dataset_config:
  binarize_labels: true
  target_value:
    segmentation:
      0: 0  # background
      1: 1  # ink
      2: 2  # horizontal fiber
      3: 3  # vertical fiber 
```

#### Multi-class with Label Remapping

Label remapping is useful when:
1. **Merging similar classes**: Combining related categories (e.g., different damage levels → any damage)
2. **Simplifying the task**: Reducing the number of classes for easier training


**Example: Combining different but similar annotations:**
```yaml
dataset_config:
  binarize_labels: true
  target_value:
    damage_segmentation:
      0: 0  # background
      1: 1  # ink
      2: 1  # medium damage  → any damage
      3: 1  # severe damage  → any damage
      4: 2  # missing/holes
```

**Example 2: aligning different annotation schemes:**
```yaml
# Original labels from different annotators:
# Annotator A used: 0=background, 1=ink, 2=faded_ink
# Annotator B used: 0=background, 1=clear_ink, 2=unclear_ink, 3=traces
dataset_config:
  binarize_labels: true
  target_value:
    ink_detection:
      0: 0  # background
      1: 1  # ink/clear_ink     → visible ink
      2: 1  # faded_ink/unclear → visible ink  
      3: 2  # traces            → barely visible
```

#### Multi-class with Region Combinations

Region combinations allow you to define new classes that represent unions of existing classes:

```yaml
dataset_config:
  binarize_labels: true
  target_value:
    tissue_segmentation:
      mapping:
        0: 0  # background
        1: 1  # healthy tissue
        2: 2  # damaged tissue
        3: 3  # scar tissue
      regions:
        4: [1, 2]      # living tissue (healthy OR damaged)
        5: [2, 3]      # abnormal tissue (damaged OR scar)
```

**Note:** Region IDs must not conflict with existing mapped class values.

### Label Values
- 0: Always background
- 1, 2, 3, ...: Different classes (mutually exclusive)

---

## Multi-task Learning

Multi-task learning trains a single model (using a shared encoder but separate decoders) to perform multiple segmentation tasks simultaneously. Tasks can overlap (e.g., a pixel can be both "ink" and "damaged").

### Napari Dataset Structure

Multiple label layers, one per task:
```
image_layer_name: "my_image"
label_layer_1: "my_image_ink"      # Task 1
label_layer_2: "my_image_damage"   # Task 2
label_layer_3: "my_image_substrate"  # Task 3
mask_layer_name: "my_image_mask"   # Optional, shared across tasks
```

**Example:**
- Image layer: `"fragment_042"`
- Label layers:
  - `"fragment_042_ink"` (ink detection)
  - `"fragment_042_damage"` (damage detection)
  - `"fragment_042_substrate"` (substrate type - multi-class)
- Mask layer: `"fragment_042_mask"` (optional)

### TIF/Zarr Dataset Structure

```
data_path/
├── images/
│   ├── image1.tif         # Single image file 
│   ├── image2.tif         # Single image file 
│   └── ...
├── labels/
│   ├── image1_ink.tif         # Binary labels for ink
│   ├── image1_damage.tif      # Binary labels for damage
│   ├── image2_ink.tif         # Binary labels for ink
│   ├── image2_damage.tif      # Binary labels for damage
│   └── ...
└── masks/          # Optional
    ├── image1_ink.tif         # Can be task-specific
    ├── image1_damage.tif      # or use same mask
    ├── image1_ink.tif         # for all tasks
    └── ...
```

**Note:** Name your images without task suffixes (e.g., `image1.tif`) and specify the task with a label suffix.

### Configuration for Binary Multi-task

```yaml
dataset_config:
  binarize_labels: true
  target_value:
    ink: 1      # Binary: contains ink
    fiber: 1    # Binary: contains fiber
    rocks: 1    # Binary: contains rocks

# The system will automatically configure targets based on your data
# Each binary task will have:
# - channels: 1
# - activation: "sigmoid"
# - loss_fn: Based on configuration or defaults
```

### Configuration for Mixed Multi-task (Binary + Multi-class)

```yaml
dataset_config:
  binarize_labels: true
  target_value:
    ink: 1              # Binary task
    damage: 1           # Binary task
    substrate_type:     # Multi-class task (simple format)
      0: 0  # background/air
      1: 1  # papyrus
      2: 2  # parchment
      3: 3  # wood
    # OR use the structured format for multi-class:
    # substrate_type:
    #   mapping:
    #     0: 0  # background/air
    #     1: 1  # papyrus
    #     2: 2  # parchment
    #     3: 3  # wood

# The system will automatically configure:
# - Binary tasks: channels=1, activation="sigmoid"
# - Multi-class tasks: channels=num_classes, activation="softmax"
```

---

## Label Value Mapping

### Understanding `binarize_labels` and `target_value`

1. **When `binarize_labels: true`**:
   - The system will remap your label values according to `target_value`
   - For binary tasks: Any non-zero pixel becomes the specified value
   - For multi-class: Values are remapped according to the dictionary

2. **When `binarize_labels: false`**:
   - Labels are used as-is without any remapping
   - Your labels must already be in the correct format (0/1 for binary, 0/1/2/... for multi-class)
   - `target_value` is ignored

### Examples of Label Remapping

**Binary Task Original Labels:**
```
Label image contains: [0, 1, 2, 5, 10, 255]
With target_value: ink: 1
Result after remapping: [0, 1, 1, 1, 1, 1]
```

**Multi-class with Merging:**
```
Original annotation scheme:
  0: background
  1: black ink
  2: brown ink  
  3: red ink
  4: blue ink
  5: damage

Simplified scheme for training:
target_value:
  ink_and_damage:
    0: 0  # background
    1: 1  # any ink color
    2: 1  # any ink color
    3: 1  # any ink color
    4: 1  # any ink color
    5: 2  # damage

Result: 3 classes instead of 6
```

---

## Validation

Use the validation script to check your data structure and configuration:

```bash
# For napari datasets
python vesuvius/models/datasets/validate_labels.py \
    --config your_config.yaml \
    --dataset napari

# For TIF datasets
python vesuvius/models/datasets/validate_labels.py \
    --config your_config.yaml \
    --dataset tif \
    --data-path /path/to/your/data

# For Zarr datasets
python vesuvius/models/datasets/validate_labels.py \
    --config your_config.yaml \
    --dataset zarr \
    --data-path /path/to/your/data
```

The validation script will:
1. Analyze the unique values in your label data
2. Check if your configuration matches the data
3. Provide recommendations for configuration
4. Report any mismatches or missing mappings

### Common Issues and Solutions

1. **"No target value configured"**
   - Add the target to your `target_value` configuration

2. **"Unexpected label values found"**
   - Your labels contain values not in the configuration
   - Either add mappings for these values or clean your data

3. **"Values found in data but not in configuration"** (multi-class)
   - Add mappings for all non-zero values in your labels

---

## Best Practices

1. **Consistent Naming**: Use descriptive, consistent task names
2. **Validate Before Training**: Always run the validation script
3. **Choose the Right Structure**:
   - Binary tasks: When detecting presence/absence
   - Multi-class: When classes are mutually exclusive
   - Multi-task: When one pixel can represent multiple classes

## Quick Reference

| Scenario | Label Structure | When to Use | Example |
|----------|----------------|-------------|---------|
| Binary | 0 = background<br>Non-zero = foreground | Detecting presence/absence | Ink detection |
| Multi-class | 0, 1, 2, 3... | Mutually exclusive categories | Material type |
| Multi-task | Multiple binary/multi-class | Multiple overlapping annotations | Ink + damage + substrate |


