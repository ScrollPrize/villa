#!/usr/bin/env python3
"""
Script to add a segmentation layer to a JSON file.
Takes a zarr path as argument and adds it as a segmentation layer to the JSON.
"""

import json
import argparse
import sys
from pathlib import Path
from copy import deepcopy


def add_segmentation_layer(json_file, zarr_base_path, layer_name="segmentation", output_file=None):
    """
    Add a segmentation layer to a JSON file.
    
    Args:
        json_file: Path to the input JSON file
        zarr_base_path: Base path to the zarr file (without /0, /1, etc.)
        layer_name: Name for the segmentation layer (default: "segmentation")
        output_file: Path to output JSON file (if None, adds _with_segmentation suffix)
    """
    
    # Read the input JSON
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Check if there are existing data layers
    if 'dataLayers' not in data or len(data['dataLayers']) == 0:
        print("Error: No existing data layers found in the JSON file")
        sys.exit(1)
    
    # Get the first existing layer as a template
    template_layer = data['dataLayers'][0]
    
    # Create a new segmentation layer based on the template
    segmentation_layer = deepcopy(template_layer)
    
    # Update the segmentation-specific fields
    segmentation_layer['name'] = layer_name
    segmentation_layer['category'] = 'segmentation'
    
    # Update the paths in mags to use the new zarr path
    # Remove trailing slash if present
    zarr_base_path = zarr_base_path.rstrip('/')
    if 'mags' in segmentation_layer:
        for i, mag in enumerate(segmentation_layer['mags']):
            # Replace the path with the new zarr path + resolution level
            mag['path'] = f"{zarr_base_path}/{i}"
    
    # Ensure coordinateTransformations field exists (present in segmentation layers)
    if 'coordinateTransformations' not in segmentation_layer:
        segmentation_layer['coordinateTransformations'] = []
    
    # Append the segmentation layer at the end of dataLayers
    data['dataLayers'].append(segmentation_layer)
    
    # Determine output file name
    if output_file is None:
        input_path = Path(json_file)
        output_file = input_path.parent / f"{input_path.stem}_with_segmentation{input_path.suffix}"
    
    # Write the modified JSON
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Successfully added segmentation layer '{layer_name}' to {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Add a segmentation layer to a JSON file using a zarr path"
    )
    parser.add_argument(
        "json_file",
        help="Path to the input JSON file"
    )
    parser.add_argument(
        "zarr_path",
        help="Base path to the zarr file (e.g., https://example.com/data.zarr)"
    )
    parser.add_argument(
        "--layer-name",
        default="segmentation",
        help="Name for the segmentation layer (default: segmentation)"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Path to output JSON file (default: adds _with_segmentation suffix)"
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.json_file).exists():
        print(f"Error: Input file '{args.json_file}' does not exist")
        sys.exit(1)
    
    # Add the segmentation layer
    add_segmentation_layer(
        args.json_file,
        args.zarr_path,
        args.layer_name,
        args.output
    )


if __name__ == "__main__":
    main()