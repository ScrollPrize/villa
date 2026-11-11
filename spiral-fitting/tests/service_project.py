"""The ``project`` block a service session request carries
(fit_session.SpiralProject): a project of one volume with its facts and an
umbilicus."""

from object_store_fixture import VOLUME

UMBILICUS = {"store": "6c1f0e4e-6d7a-4d55-9d0c-2b8f9a0c1e12", "uuid": "6c1f0e4e-6d7a-4d55-9d0c-2b8f9a0c1e11",
             "generation": "a7d5c3e0-1b2c-4d3e-8f90-0a1b2c3d4e5f", "version": 3,
             "points_zyx": [[0.0, 32.0, 32.0], [64.0, 32.0, 32.0]]}


# A volume the fixture datasets' scrolls describe: 9.6 µm voxels and a
# catalog orientation that makes the spiral wind outwards clockwise.
FACTS = {"voxel_size_um": 9.6, "z_direction_is_top_to_bottom": False,
         "left_handed_coordinates": False}


SELECTED_UMBILICUS = {"reference": {"store": UMBILICUS["store"], "generation": UMBILICUS["generation"],
                                    "object": UMBILICUS["uuid"], "version": UMBILICUS["version"]},
                      "points_zyx": UMBILICUS["points_zyx"]}


def project(volume=VOLUME, *, name="s1", umbilicus=SELECTED_UMBILICUS, lasagna=None,
            fiber_inference=None, **facts):
    entry = {**FACTS, **facts}
    entry = {key: value for key, value in entry.items() if value is not None}
    if umbilicus is not None:
        entry["umbilicus"] = umbilicus
    block = {"name": name, "volumes": {volume: entry}}
    if lasagna is not None:
        block["lasagna"] = lasagna
    if fiber_inference is not None:
        block["fiber_inference"] = fiber_inference
    return block


def with_project(request, **kwargs):
    """``request`` with the project block (``project``'s arguments)."""
    return {**request, "project": project(**kwargs)}
