#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

def rename_files(folder, suffix, mode):
    """
    Add or remove suffix from files in a folder.
    
    Args:
        folder: Path to the folder containing files
        suffix: The suffix to add or remove
        mode: 'add' or 'remove'
    """
    folder_path = Path(folder)
    
    if not folder_path.exists():
        print(f"Error: Folder '{folder}' does not exist.")
        return 1
    
    if not folder_path.is_dir():
        print(f"Error: '{folder}' is not a directory.")
        return 1
    
    # Get all files in the folder (not subdirectories)
    files = [f for f in folder_path.iterdir() if f.is_file()]
    
    if not files:
        print(f"No files found in '{folder}'.")
        return 0
    
    renamed_count = 0
    
    for file_path in files:
        stem = file_path.stem
        extension = file_path.suffix
        
        if mode == 'add':
            # Check if suffix already exists
            if stem.endswith(suffix):
                print(f"Skipping '{file_path.name}': already has suffix '{suffix}'")
                continue
            
            new_name = f"{stem}{suffix}{extension}"
        else:  # mode == 'remove'
            # Check if suffix exists
            if not stem.endswith(suffix):
                print(f"Skipping '{file_path.name}': doesn't have suffix '{suffix}'")
                continue
            
            new_name = f"{stem[:-len(suffix)]}{extension}"
        
        new_path = file_path.parent / new_name
        
        # Check if target file already exists
        if new_path.exists():
            print(f"Warning: Cannot rename '{file_path.name}' to '{new_name}': target already exists")
            continue
        
        try:
            file_path.rename(new_path)
            print(f"Renamed: '{file_path.name}' -> '{new_name}'")
            renamed_count += 1
        except Exception as e:
            print(f"Error renaming '{file_path.name}': {e}")
    
    print(f"\nTotal files renamed: {renamed_count}")
    return 0

def main():
    parser = argparse.ArgumentParser(
        description="Add or remove a suffix from files in a folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add suffix '_backup' to all files in /path/to/folder
  %(prog)s -m add -f /path/to/folder -s _backup
  
  # Remove suffix '_old' from all files in current directory
  %(prog)s --mode remove --folder . --suffix _old
"""
    )
    
    parser.add_argument(
        '-m', '--mode',
        required=True,
        choices=['add', 'remove'],
        help="Mode: 'add' to add suffix, 'remove' to remove suffix"
    )
    
    parser.add_argument(
        '-f', '--folder',
        required=True,
        help="Path to the folder containing files"
    )
    
    parser.add_argument(
        '-s', '--suffix',
        required=True,
        help="The suffix to add or remove (e.g., '_backup', '_v2')"
    )
    
    args = parser.parse_args()
    
    return rename_files(args.folder, args.suffix, args.mode)

if __name__ == "__main__":
    sys.exit(main())