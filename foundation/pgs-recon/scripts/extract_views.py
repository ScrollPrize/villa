"""
Example script showing how to remove specific views from a calibrated MVG scene

Usage:

# Convert a calibrated scene to JSON
openMVG_main_ConvertSfM_DataFormat -i sfm_data.bin -o sfm_data.json

# Extract a subset of views
python3 extract_views.py -i sfm_data.json -o sfm_filtered.json --include path/to/subset/image*.jpg --root-path path/to/subset

# Convert back to binary to save space
openMVG_main_ConvertSfM_DataFormat -i sfm_filtered.json -o sfm_filtered.bin

### optional: Recompute features, matches, and scene structure if adding/removing image masks ###
openMVG_main_ComputeFeatures -i sfm_filtered.bin -o filtered_matches [optional flags]
openMVG_main_ComputeMatches -i sfm_filtered.bin -o filtered_matches/matches.bin [optional flags]
openMVG_main_ComputeStructureFromKnownPoses -i sfm_filtered.bin -o sfm_filtered.bin -m filtered_matches -d -f filtered_matches/matches.bin --bundle_adjustment 1
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-scene', type=Path, required=True,
                        help='Input scene file')
    parser.add_argument('-o', '--output-scene', type=Path, required=True,
                        help='Output scene file')
    parser.add_argument('--include', nargs='+', type=Path, required=True,
                        help='List of images to include in the final scene. '
                             'Only the filename will be considered.')
    parser.add_argument('--root-path', type=Path,
                        help='If provided, replace the root path in the scene '
                             'file. Provided path will be converted to an '
                             'absolute path.')
    args = parser.parse_args()

    # get the unique list of files
    include_list = {str(a.name) for a in args.include}

    # open the data
    print('opening the input scene...')
    with open(args.input_scene) as f:
        data = json.load(f)

    # map old IDs -> new IDs
    view_map = {}
    intrinsic_map = {}
    pose_map = {}

    # filter views
    views = []
    print('filtering views...')
    for v in data['views']:
        img_file = v['value']['ptr_wrapper']['data']['filename']
        if img_file in include_list:
            # original view key
            id_view = v['key']
            # update with new view key
            view_map[id_view] = len(view_map)
            v['key'] = view_map[id_view]
            v['value']['ptr_wrapper']['data']['id_view'] = view_map[id_view]

            # get linked IDs
            id_intr = v['value']['ptr_wrapper']['data']['id_intrinsic']
            if id_intr not in intrinsic_map.keys():
                intrinsic_map[id_intr] = len(intrinsic_map)
            v['value']['ptr_wrapper']['data']['id_intrinsic'] = intrinsic_map[
                id_intr]

            id_pose = v['value']['ptr_wrapper']['data']['id_pose']
            if id_pose not in pose_map.keys():
                pose_map[id_pose] = len(pose_map)
            v['value']['ptr_wrapper']['data']['id_pose'] = pose_map[id_pose]

            # add updated view to data
            views.append(v)

    # filter intrinsics
    intrinsics = []
    print('filtering intrinsics...')
    for i in data['intrinsics']:
        if i['key'] in intrinsic_map.keys():
            i['key'] = intrinsic_map[i['key']]
            if i['value']['polymorphic_id'] < 2_000_000_000:
                id_ref = i['value']['polymorphic_id']
                orig_value = data['intrinsics'][id_ref - 1]['value']
                i['value']['polymorphic_id'] = orig_value['polymorphic_id']
                i['value']['polymorphic_name'] = orig_value['polymorphic_name']
            intrinsics.append(i)

    # filter poses
    poses = []
    print('filtering poses...')
    for p in data['extrinsics']:
        if p['key'] in pose_map.keys():
            p['key'] = pose_map[p['key']]
            poses.append(p)

    # filter structure points
    structure = []
    print('filtering structure...')
    for s in data['structure']:
        observations = []
        for o in s['value']['observations']:
            if o['key'] in view_map.keys():
                # replace the view key
                o['key'] = view_map[o['key']]
                observations.append(o)
        s['value']['observations'] = observations
        if len(observations):
            s['key'] = len(structure)
            structure.append(s)

    # update the data structure
    if args.root_path is not None:
        data['root_path'] = str(args.root_path.resolve())
    data['views'] = views
    data['intrinsics'] = intrinsics
    data['extrinsics'] = poses
    data['structure'] = structure

    # save the output scene
    print('saving the filtered scene...')
    with open(args.output_scene, 'w') as f:
        json.dump(data, f, indent=4)

    print('done.')


if __name__ == '__main__':
    main()
