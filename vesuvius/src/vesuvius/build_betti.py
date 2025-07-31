#!/usr/bin/env python3
"""
Manually build Betti-Matching-3D for Vesuvius
"""

import subprocess
import sys
from pathlib import Path
import platform

def build_betti_matching():
    """Clone and build Betti-Matching-3D"""
    # Find the vesuvius root directory
    vesuvius_root = Path(__file__).parent.parent.parent.resolve()
    betti_dir = vesuvius_root / "external" / "Betti-Matching-3D"
    
    print(f"Vesuvius root: {vesuvius_root}")
    print(f"Betti directory will be: {betti_dir}")
    
    try:
        # Create external directory if it doesn't exist
        betti_dir.parent.mkdir(parents=True, exist_ok=True)
        
        # Clone if not exists
        if not betti_dir.exists():
            print("\nCloning Betti-Matching-3D...")
            subprocess.run([
                "git", "clone", 
                "https://github.com/nstucki/Betti-Matching-3D.git",
                str(betti_dir)
            ], check=True)
        else:
            print(f"\nBetti-Matching-3D already exists at {betti_dir}")
        
        # Build
        build_dir = betti_dir / "build"
        build_dir.mkdir(exist_ok=True)
        
        print(f"\nBuilding Betti-Matching-3D in {build_dir}...")
        
        # Configure with CMake
        cmake_cmd = ["cmake", ".."]
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            cmake_cmd = ["cmake", "-DCMAKE_OSX_ARCHITECTURES=arm64", ".."]
        
        print(f"Running: {' '.join(cmake_cmd)}")
        subprocess.run(cmake_cmd, cwd=build_dir, check=True)
        
        # Build
        print("\nRunning: make")
        subprocess.run(["make"], cwd=build_dir, check=True)
        
        print("\nBetti-Matching-3D built successfully!")
        print(f"Build location: {build_dir}")
        
        # Verify the build
        if (build_dir / "betti_matching.so").exists() or \
           (build_dir / "betti_matching.cpython-*.so").exists() or \
           list(build_dir.glob("betti_matching*.so")):
            print("✓ Found betti_matching module")
        else:
            print("⚠ Warning: Could not find betti_matching.so in build directory")
            print("Contents of build directory:")
            for f in build_dir.iterdir():
                print(f"  - {f.name}")
        
    except subprocess.CalledProcessError as e:
        print(f"\nFailed to build Betti-Matching-3D: {e}")
        print(f"You may need to build it manually:")
        print(f"  cd {betti_dir}")
        print(f"  mkdir -p build && cd build")
        print(f"  cmake ..")
        print(f"  make")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    build_betti_matching()