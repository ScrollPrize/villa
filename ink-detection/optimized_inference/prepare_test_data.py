#!/usr/bin/env python3
"""
Helper script to prepare test data for local inference testing.
Can create dummy test data or help organize existing layer files.
"""
import os
import argparse
import shutil
from pathlib import Path
from typing import List
import numpy as np
import cv2

def create_dummy_layers(output_dir: Path, 
                       num_layers: int = 26,
                       height: int = 512, 
                       width: int = 512,
                       add_noise: bool = True) -> None:
    """
    Create dummy layer files for testing.
    
    Args:
        output_dir: Directory to save layers
        num_layers: Number of layers to create
        height: Image height
        width: Image width
        add_noise: Whether to add noise to make layers look realistic
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating {num_layers} dummy layers in {output_dir}")
    print(f"Layer dimensions: {height} x {width}")
    
    for i in range(num_layers):
        # Create a base image with some pattern
        img = np.zeros((height, width), dtype=np.uint8)
        
        # Add some geometric patterns to simulate scroll layers
        center_x, center_y = width // 2, height // 2
        
        # Add some circles and lines
        cv2.circle(img, (center_x, center_y), min(height, width) // 4, 100, 2)
        cv2.line(img, (0, center_y), (width, center_y), 50, 1)
        cv2.line(img, (center_x, 0), (center_x, height), 50, 1)
        
        # Add some text-like patterns
        for j in range(5):
            y = center_y + (j - 2) * 40
            if 0 < y < height:
                cv2.rectangle(img, (center_x - 100, y - 5), (center_x + 100, y + 5), 80, -1)
        
        # Add noise to make it look more realistic
        if add_noise:
            noise = np.random.normal(0, 20, (height, width)).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Add some variation based on layer index
        brightness_offset = int(30 * np.sin(i * 0.2))
        img = np.clip(img.astype(np.int16) + brightness_offset, 0, 255).astype(np.uint8)
        
        # Save layer
        layer_path = output_dir / f"{i:02d}.tif"
        cv2.imwrite(str(layer_path), img)
        
        if i % 5 == 0:
            print(f"  Created layer {i:02d}.tif")
    
    print(f"✅ Created {num_layers} dummy layers successfully")

def create_dummy_mask(output_dir: Path, 
                     height: int = 512, 
                     width: int = 512,
                     mask_name: str = "mask.png") -> None:
    """
    Create a dummy fragment mask.
    
    Args:
        output_dir: Directory to save mask
        height: Mask height
        width: Mask width
        mask_name: Mask filename
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a circular mask with some irregular edges
    mask = np.zeros((height, width), dtype=np.uint8)
    center_x, center_y = width // 2, height // 2
    radius = min(height, width) // 3
    
    # Create circular base
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    
    # Add some irregular edges
    for angle in range(0, 360, 10):
        angle_rad = np.radians(angle)
        r_variation = radius + int(20 * np.sin(angle * 0.1))
        x = int(center_x + r_variation * np.cos(angle_rad))
        y = int(center_y + r_variation * np.sin(angle_rad))
        cv2.circle(mask, (x, y), 15, 255, -1)
    
    mask_path = output_dir / mask_name
    cv2.imwrite(str(mask_path), mask)
    print(f"✅ Created dummy mask: {mask_path}")

def organize_existing_layers(source_dir: Path, 
                           output_dir: Path,
                           file_pattern: str = "*") -> None:
    """
    Organize existing layer files into the expected structure.
    
    Args:
        source_dir: Directory containing existing layer files
        output_dir: Output directory for organized layers
        file_pattern: Pattern to match files
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
    
    # Find all image files
    image_extensions = ['.tif', '.tiff', '.jpg', '.jpeg', '.png']
    files = []
    
    for ext in image_extensions:
        files.extend(source_dir.glob(f"*{ext}"))
        files.extend(source_dir.glob(f"*{ext.upper()}"))
    
    files.sort()
    
    if not files:
        print(f"⚠️  No image files found in {source_dir}")
        return
    
    print(f"Found {len(files)} image files in {source_dir}")
    print("Organizing files...")
    
    for i, file_path in enumerate(files):
        # Create standardized filename
        new_filename = f"{i:02d}.tif"
        new_path = output_dir / new_filename
        
        # Copy file
        shutil.copy2(file_path, new_path)
        print(f"  {file_path.name} -> {new_filename}")
    
    print(f"✅ Organized {len(files)} files in {output_dir}")

def print_usage_examples():
    """Print usage examples for the test script."""
    print("\n📖 Usage Examples:")
    print("=" * 50)
    
    print("\n1. Basic test with dummy data:")
    print("   python test_local_inference.py ./test_layers/")
    
    print("\n2. Test with custom layer range:")
    print("   python test_local_inference.py ./test_layers/ --layer-start 5 --layer-end 20")
    
    print("\n3. Test with mask:")
    print("   python test_local_inference.py ./test_layers/ --mask-path ./test_layers/mask.png")
    
    print("\n4. Test with custom model:")
    print("   python test_local_inference.py ./test_layers/ --model-path /path/to/your/model.ckpt")
    
    print("\n5. Save results:")
    print("   python test_local_inference.py ./test_layers/ --output-path ./results/prediction")
    
    print("\n6. Force reverse segment:")
    print("   python test_local_inference.py ./test_layers/ --force-reverse")
    
    print("\n7. Verbose output:")
    print("   python test_local_inference.py ./test_layers/ --verbose")
    
    print("\n8. Use CPU instead of GPU:")
    print("   python test_local_inference.py ./test_layers/ --device cpu")

def main():
    parser = argparse.ArgumentParser(description="Prepare test data for local inference")
    parser.add_argument("--create-dummy", metavar="DIR", help="Create dummy test layers in specified directory")
    parser.add_argument("--organize", nargs=2, metavar=("SOURCE", "OUTPUT"), 
                       help="Organize existing layers from SOURCE to OUTPUT directory")
    parser.add_argument("--num-layers", type=int, default=26, help="Number of dummy layers to create")
    parser.add_argument("--height", type=int, default=512, help="Layer height")
    parser.add_argument("--width", type=int, default=512, help="Layer width")
    parser.add_argument("--with-mask", action="store_true", help="Create a dummy mask file")
    parser.add_argument("--examples", action="store_true", help="Show usage examples")
    
    args = parser.parse_args()
    
    print("🛠️  Test Data Preparation Tool")
    print("=" * 40)
    
    if args.examples:
        print_usage_examples()
        return
    
    if args.create_dummy:
        output_dir = Path(args.create_dummy)
        print(f"Creating dummy test data in {output_dir}")
        
        create_dummy_layers(
            output_dir,
            num_layers=args.num_layers,
            height=args.height,
            width=args.width
        )
        
        if args.with_mask:
            create_dummy_mask(output_dir, args.height, args.width)
        
        print(f"\n🎉 Test data created successfully!")
        print(f"   Location: {output_dir}")
        print(f"   Layers: {args.num_layers} files ({args.height}x{args.width})")
        print(f"   Mask: {'Yes' if args.with_mask else 'No'}")
        print(f"\n💡 Next steps:")
        print(f"   cd {Path(__file__).parent}")
        print(f"   python test_local_inference.py {output_dir}")
    
    elif args.organize:
        source_dir, output_dir = args.organize
        print(f"Organizing layers from {source_dir} to {output_dir}")
        
        organize_existing_layers(Path(source_dir), Path(output_dir))
        
        if args.with_mask:
            # Try to find existing mask
            mask_files = list(Path(source_dir).glob("*mask*"))
            if mask_files:
                mask_file = mask_files[0]
                shutil.copy2(mask_file, Path(output_dir) / "mask.png")
                print(f"✅ Copied mask: {mask_file.name} -> mask.png")
            else:
                create_dummy_mask(Path(output_dir), args.height, args.width)
        
        print(f"\n💡 Next steps:")
        print(f"   cd {Path(__file__).parent}")
        print(f"   python test_local_inference.py {output_dir}")
    
    else:
        print("No action specified. Use --help for options or --examples for usage examples.")
        print("\n💡 Quick start:")
        print("   python prepare_test_data.py --create-dummy ./test_layers --with-mask")
        print("   python test_local_inference.py ./test_layers")

if __name__ == "__main__":
    main()