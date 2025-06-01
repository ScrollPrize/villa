#!/usr/bin/env python3
"""
Test script to verify that labels from gui_main are properly displayed in ome_zarr_view.

Usage:
1. Run this script: python test_ome_zarr_labels.py
2. Open the OME-Zarr view from the View menu
3. Label some points in the main GUI (gui_main)
4. Check the console output - you should see debug messages showing:
   - Labels being sent from gui_main to ome_zarr_view
   - Windings being computed in the worker
   - Brushes being updated in the view
5. The labeled points should appear colored in the OME-Zarr view
"""

import sys
import os
sys.path.append('../ThaumatoAnakalyptor/graph_problem/build')

from gui_main import PointCloudLabeler
from PyQt5.QtWidgets import QApplication

def main():
    app = QApplication(sys.argv)
    
    # Create the main labeler window
    labeler = PointCloudLabeler()
    labeler.show()
    
    print("=" * 80)
    print("OME-Zarr Label Display Test")
    print("=" * 80)
    print("\nInstructions:")
    print("1. Open the OME-Zarr view from View -> Open OME-Zarr View")
    print("2. Label some points in the main GUI")
    print("3. Watch the console for debug output")
    print("4. Verify that labeled points appear colored in the OME-Zarr view")
    print("\nExpected debug output:")
    print("- [ome_zarr_view] update_overlay_labels called...")
    print("- [_trigger_overlay_labels_update] Requesting update...")
    print("- [Worker compute_labels_xy] Processing...")
    print("- [on_overlay_labels_computed] Received windings...")
    print("- [get_brushes] Valid windings...")
    print("\nIf you see labels in gui_main but no colors in ome_zarr_view:")
    print("- Check if 'Valid windings' shows 0%")
    print("- Check if worker is processing the correct number of nodes")
    print("=" * 80)
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 