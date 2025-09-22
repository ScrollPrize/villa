"""Public utility surface for the Vesuvius package."""

from vesuvius.utils.catalog import (
    collect_subfolders,
    is_aws_ec2_instance,
    list_cubes,
    list_files,
    scrape_website,
    update_list,
)

__all__ = [
    "collect_subfolders",
    "is_aws_ec2_instance",
    "list_cubes",
    "list_files",
    "scrape_website",
    "update_list",
]
