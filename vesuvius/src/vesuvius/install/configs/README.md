# Vesuvius Configuration Files

This directory contains configuration files used by the Vesuvius package to access scroll data and annotated cubes.

## Files

- `scrolls.yaml`: Contains URLs for accessing scroll volumes and segments
- `cubes.yaml`: Contains URLs for accessing annotated 3D cubes with ink labels

## Usage

These configuration files are used by the `Volume` and `Cube` classes to locate data. When you install the package, these config files are automatically included.

### Adding Your Own Data Sources

You can modify these files to add your own data sources or update existing ones. Follow the structure in the example files:

#### For scrolls.yaml:

```yaml
"scroll_id":
  "energy":
    "resolution":
      volume: "url_to_volume"
      segments:
        "segment_id": "url_to_segment"
```

#### For cubes.yaml:

```yaml
scroll_id:
  energy:
    resolution:
      ZZZZZ_YYYYY_XXXXX: "url_to_cube"
```

## Finding Configuration Files

If you're using the package in development mode (installed with `pip install -e .`), the config files will be located in:

```
/path/to/vesuvius/install/configs/
```

If you're using the package installed from PyPI, the config files will be located in:

```
/path/to/site-packages/vesuvius/install/configs/
```

You can also provide your own configuration files by specifying the file paths when initializing the `Volume` or `Cube` objects.

## Troubleshooting

If you encounter a `FileNotFoundError` when trying to access data, make sure:

1. The package is properly installed
2. The configuration files exist and are correctly formatted
3. You have internet access if trying to reach remote data sources

For more information, visit: https://github.com/ScrollPrize/villa
