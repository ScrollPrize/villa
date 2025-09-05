import json
import click
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict


@click.command()
@click.argument('input_folder', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument('output_file', type=click.Path())
def main(input_folder: str, output_file: str):
    """
    Extract seed coordinates from patch folders and group by vc_gsfs_mode.
    
    INPUT_FOLDER: Path to folder containing patch subfolders with meta.json files
    OUTPUT_FILE: Path to output JSON file where grouped seeds will be saved
    """
    
    seeds_by_mode: defaultdict[str, List[Tuple[float]]] = defaultdict(list)
    
    patch_count = 0
    processed_count = 0
    
    for patch_dir in Path(input_folder).iterdir():
        if not patch_dir.is_dir():
            continue
            
        patch_count += 1
        
        try:

            with open(patch_dir / "meta.json", 'r') as f:
                meta_data = json.load(f)
            
            seed = meta_data.get('seed')
            mode = meta_data.get('vc_gsfs_mode')
            
            if seed is None or mode is None or not isinstance(seed, list) or len(seed) != 3:
                click.echo(f"Warning: Invalid seed or mode in {patch_dir.name}/meta.json", err=True)
                continue
                
            seeds_by_mode[mode].append(tuple(seed))
            processed_count += 1
            if processed_count % 100 == 0:
                click.echo(f"Processed {processed_count} patches")
            
        except json.JSONDecodeError as e:
            click.echo(f"Error: Failed to parse JSON in {patch_dir.name}/meta.json: {e}", err=True)
        except Exception as e:
            click.echo(f"Error: Failed to process {patch_dir.name}/meta.json: {e}", err=True)
    
    click.echo(f"\nSummary:")
    click.echo(f"Total patch directories found: {patch_count}")
    click.echo(f"Successfully processed: {processed_count}")
    for mode in seeds_by_mode:
        click.echo(f"Seeds with mode '{mode}': {len(seeds_by_mode[mode])}")
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(seeds_by_mode, f, indent=2)
        

if __name__ == '__main__':
    main()
