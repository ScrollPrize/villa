### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius Challenge 2023

from .instances_to_sheets import display_main_sheet, load_main_sheet
import numpy as np

if __name__ == '__main__':
    print(int(-0.5))
    print(np.array([0.5, -0.5]).astype(int))
    path_base = "/media/julian/SSD4TB/scroll3_surface_points/"
    path_ta = "point_cloud_colorized_verso_subvolume_main_sheet.ta"
    point_cloud_name = "point_cloud_colorized_verso_subvolume_blocks"
    import argparse
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Display ThaumatoAnakalyptor Papyrus Sheets')
    parser.add_argument('--path_base', type=str, help='Base path for instances (.tar) and main sheet (.ta)', default=path_base)
    parser.add_argument('--path_ta', type=str, help='Papyrus sheet under path_base (with custom .ta ending)', default=path_ta)
    parser.add_argument('--point_cloud_name', type=str, help='Name of pointcloud blocks', default=point_cloud_name)
    
    # Take arguments back over
    args = parser.parse_args()
    path_base = args.path_base
    path_ta = args.path_ta
    point_cloud_name = args.point_cloud_name

    path = path_base + point_cloud_name
    path_ta = path_base + path_ta

    print(f"Loading main sheet from {path_ta} from instances path {path}")
    
    sample_ratio = 0.5
    main_sheet, _ = load_main_sheet(path=path, path_ta=path_ta, sample_ratio=sample_ratio) # main_sheet_fused
    nr_patches = 0
    for volume_id in main_sheet:
        if len(main_sheet[volume_id]) > 1:
            print(f"Volume {volume_id} has {len(main_sheet[volume_id])} patches")
        for patch_id in main_sheet[volume_id]:
            nr_patches += 1
    print(f"Main Sheet has {nr_patches} patches")
    display_main_sheet(main_sheet, nr_points_show=10000, show_normals=False)