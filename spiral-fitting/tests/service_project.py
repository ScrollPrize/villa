"""The ``project`` block a service session request carries
(fit_session.SpiralProject): a project of one volume with an umbilicus."""

from object_store_fixture import VOLUME

UMBILICUS = {"uuid": "6c1f0e4e-6d7a-4d55-9d0c-2b8f9a0c1e11",
             "revision": "a7d5c3e0-1b2c-4d3e-8f90-0a1b2c3d4e5f",
             "points_zyx": [[0.0, 32.0, 32.0], [64.0, 32.0, 32.0]]}


def project(volume=VOLUME, *, name="s1", umbilicus=UMBILICUS, **facts):
    entry = dict(facts)
    if umbilicus is not None:
        entry["umbilicus"] = umbilicus
    return {"name": name, "volumes": {volume: entry}}


def with_project(request, **kwargs):
    """``request`` with the project block (``project``'s arguments)."""
    return {**request, "project": project(**kwargs)}
