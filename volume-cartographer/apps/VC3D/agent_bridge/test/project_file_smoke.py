#!/usr/bin/env python3
"""Exercise standalone projects against the real VC3D bridge, without network data."""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from bridge_client import BridgeClient, BridgeError
from smoke_offscreen import create_test_volume, launch, unique_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vc3d", required=True)
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="vcproj-smoke-") as temporary:
        root = Path(temporary)
        config = root / "config"
        config.mkdir()
        (config / "VC3D.ini").write_text(
            "[viewer]\nusername=project-smoke\nremote_cache_dir=" + str(root / "cache") +
            "\n[project]\nshow_open_data_catalog_on_startup=false\nauto_open=0\n")
        volume = create_test_volume(root / "volumes" / "scan", "scan")
        process = launch(args.vc3d, unique_name("projects"), config_home=str(config))
        client = None
        try:
            client = BridgeClient(process.wait_for_handshake(120))

            def call(method, params=None):
                return client.call(method, params or {}, timeout=60)[0]

            kinds = ("surface", "fiber", "point_collection", "umbilicus", "volume_transform")
            stores = []
            for kind in kinds:
                result = call("project.create", {"path": str(root / kind), "volume": str(volume)})
                assert result["path"].endswith(".vcproj")
                assert Path(result["store"]).is_file()
                document = json.loads(Path(result["path"]).read_text())
                stores.append(document["stores"][0])

            project = root / "workspace.vcproj"
            document.update(name="Multiple stores", stores=stores,
                            selections={kind: store["id"] for kind, store in zip(kinds, stores)})
            document["stores"][1]["read_only"] = True
            document["stores"].append(dict(id="missing", location="missing.vcobj", read_only=False))
            document["selections"]["umbilicus"] = "missing"
            project.write_text(json.dumps(document))
            call("volume.open", {"path": str(project)})
            state = call("state.get")
            assert state["project"]["path"] == str(project)
            assert state["volume"]["id"] == "scan"
            assert len(state["stores"]) == 6
            assert state["routes"] == document["selections"]
            by_id = {store["id"]: store for store in state["stores"]}
            assert not by_id["missing"]["available"]
            assert not by_id[stores[1]["id"]]["writable"]
            assert by_id[stores[0]["id"]]["writable"]
            assert json.loads(project.read_text())["selections"]["umbilicus"] == "missing"
            call("workspace.switch", {"name": "project"})
            if args.screenshot:
                call("screenshot.capture", {"target": "window", "filePath": str(args.screenshot.resolve())})

            # An old store is not a project; a failed open preserves the workspace.
            store_path = root / stores[0]["location"]
            try:
                call("volume.open", {"path": str(store_path)})
                raise AssertionError("A .vcobj was accepted as a project")
            except BridgeError:
                pass
            assert call("state.get")["project"]["path"] == str(project)

            # Reusing a store does not import resources or settings from another project.
            shared = root / "shared.vcproj"
            call("project.create", {"path": str(shared), "volume": str(volume), "store": str(store_path)})
            call("volume.open", {"path": str(shared)})
            shared_state = call("state.get")
            assert len(shared_state["stores"]) == 1
            assert shared_state["stores"][0]["storeUuid"] == by_id[stores[0]["id"]]["storeUuid"]
            assert json.loads(project.read_text())["selections"] == document["selections"]

            # Overwriting a description preserves the referenced store.
            call("project.create", {"path": str(project), "volume": str(volume),
                                    "store": str(store_path), "overwrite": True})
            assert store_path.is_file()
            call("volume.open", {"path": str(project)})
            assert call("state.get")["stores"][0]["storeUuid"] == shared_state["stores"][0]["storeUuid"]

            stores_and_resources(call, root, volume)

            # File permissions and store permissions are independent.
            call("volume.open", {"path": str(shared)})
            project.chmod(0o444)
            try:
                call("volume.open", {"path": str(project)})
                state = call("state.get")
                assert not state["project"]["writable"]
                assert state["stores"][0]["writable"]
            finally:
                project.chmod(0o644)
            print("Project-file smoke: passed")
        except BaseException:
            print("\n".join(process.tail_log(80)))
            raise
        finally:
            if client:
                client.close()
            process.terminate()


def expect_error(call, method, params, code):
    try:
        call(method, params)
    except BridgeError as error:
        assert error.code == code, (method, error)
        return error
    raise AssertionError(f"{method} must fail with {code}")


def wait_job(call, started):
    deadline = time.monotonic() + 120
    while (status := call("job.status", {"jobId": started["jobId"]}))["state"] == "running":
        assert time.monotonic() < deadline, status
        time.sleep(0.1)
    return status


def create_array(path: Path) -> None:
    from vc_reader.reader import ZarrWriter

    path.mkdir(parents=True)
    ZarrWriter.create(path, shape=(16, 16, 16), chunks=(16, 16, 16), dtype="|u1", compressor=None)


def stores_and_resources(call, root: Path, volume: Path) -> None:
    """A project without a volume gains stores, routes, resources and an umbilicus
    through the bridge, each persisted across reopening."""
    created = call("project.create", {"path": str(root / "built")})
    project = Path(created["path"])
    assert created["volume"] is None, created
    call("volume.open", {"path": str(project)})
    state = call("state.get")
    assert state["volume"] is None and state["project"]["entries"] == [], state["project"]
    assert state["author"] == "project-smoke" and state["authorConfigured"], state
    main = state["stores"][0]["id"]
    reopen = lambda: (call("volume.open", {"path": str(project)}), call("state.get"))[1]

    # A volume attached later is selectable.
    status = wait_job(call, call("volume.attach", {"location": str(volume)}))
    assert status["state"] == "succeeded", status
    call("volume.select", {"volumeId": "scan"})

    # Stores: create, attach (idempotently, or read-only), route one kind, detach.
    extra = call("stores.create", {"path": str(root / "extra")})
    assert not extra["readOnly"] and extra["path"].endswith("extra.vcobj"), extra
    assert call("stores.attach", {"path": extra["path"], "readOnly": True})["storeId"] == extra["storeId"]
    expect_error(call, "stores.create", {"path": extra["path"]}, -32005)
    spare = call("stores.create", {"path": str(root / "spare.vcobj")})
    call("stores.detach", {"storeId": spare["storeId"]})
    assert Path(spare["path"]).is_file()
    readonly = call("stores.attach", {"path": spare["path"], "readOnly": True})
    assert readonly["readOnly"], readonly
    routed = call("stores.route", {"kind": "surface", "storeId": extra["storeId"]})
    assert routed["routes"]["surface"] == extra["storeId"], routed
    expect_error(call, "stores.route", {"kind": "surface", "storeId": "unknown"}, -32007)
    expect_error(call, "stores.route", {"kind": "track", "storeId": main}, -32602)
    assert call("state.get")["routes"]["surface"] == extra["storeId"]
    state = reopen()
    assert state["routes"]["surface"] == extra["storeId"] and state["routes"]["fiber"] == main, state["routes"]
    assert {store["id"] for store in state["stores"]} == {main, extra["storeId"], readonly["storeId"]}
    expect_error(call, "objects.import", {"kind": "surface", "path": str(root), "store": readonly["storeId"]}, -32009)
    call("stores.route", {"kind": "umbilicus", "storeId": None})
    assert reopen()["routes"]["umbilicus"] is None
    call("stores.route", {"kind": "umbilicus", "storeId": main})

    # Save As shares the stores and reopens the copy; it never overwrites.
    saved = call("project.save_as", {"path": str(root / "built-copy")})
    assert saved["path"].endswith("built-copy.vcproj") and saved["previousPath"] == str(project), saved
    copy = call("state.get")
    assert copy["project"]["path"] == saved["path"] and copy["volume"]["id"] == "scan", copy
    assert copy["routes"] == state["routes"] | {"umbilicus": main}, copy["routes"]
    expect_error(call, "project.save_as", {"path": saved["path"]}, -32005)
    call("volume.open", {"path": str(project)})

    # Resources: a normal grid and Lasagna data in each role, then detaching entries.
    grid = root / "normal_grids"
    for plane in ("xy", "xz", "yz"):
        (grid / plane).mkdir(parents=True)
    (grid / "metadata.json").write_text("{}")
    assert call("project.attach_normal_grid", {"location": str(grid)})["attached"]
    assert call("project.attach_normal_grid", {"location": str(grid)})["alreadyAttached"]
    expect_error(call, "project.attach_normal_grid", {"location": str(root / "volumes")}, -32007)
    datasets = {}
    for role, channels in (("regular", ("grad_mag", "nx", "ny")), ("fiber", ("presence", "nx", "ny"))):
        directory = root / f"lasagna-{role}"
        create_array(directory / "pred.zarr")
        manifest = directory / f"{role}.lasagna.json"
        manifest.write_text(json.dumps(dict(version=2, source_to_base=1, groups={
            channel: dict(zarr="pred.zarr", channels=[channel]) for channel in channels})))
        status = wait_job(call, call("project.attach_lasagna_manifest", {"location": str(manifest), "role": role}))
        assert status["state"] == "succeeded" and status["result"]["role"] == role, status
        datasets[role] = str(manifest)
    status = wait_job(call, call("project.attach_lasagna_manifest", {"location": datasets["fiber"], "role": "regular"}))
    assert status["state"] == "failed" and "normal" in status["message"], status
    # Entries report the locations the project persists (relative to it where possible).
    entries = {str((project.parent / entry["location"]).resolve()): entry for entry in reopen()["project"]["entries"]}
    assert entries[datasets["regular"]]["kind"] == "lasagna", entries
    assert entries[datasets["fiber"]]["kind"] == "fiber_lasagna", entries
    assert entries[str(grid)]["kind"] == "normal_grid" and entries[str(grid)]["scope"] == "everyone", entries
    grid_location = entries[str(grid)]["location"]
    assert call("project.detach_entry", {"location": grid_location}) == {"location": grid_location, "scope": "everyone"}
    expect_error(call, "project.detach_entry", {"location": grid_location}, -32007)
    assert grid.is_dir()

    # The umbilicus of the current volume: stated frames import as is, others are confirmed.
    garbage = root / "not-umbilicus.json"
    garbage.write_text("{")
    error = expect_error(call, "umbilicus.import", {"path": str(garbage)}, -32602)
    assert error.data["param"] == "path", error.data
    stamped = root / "umbilicus.json"
    stamped.write_text(json.dumps(dict(points=[[z, 8, 8] for z in range(16)], metadata=dict(
        voxelsize_um=1.0, volume_width=16, volume_height=16, volume_slices=16))))
    imported = call("umbilicus.import", {"path": str(stamped)})
    assert imported["store"] == main and not imported["replaced"], imported
    assert call("state.get")["umbilicus"]["uuids"] == [imported["uuid"]]
    expect_error(call, "umbilicus.import", {"path": str(stamped)}, -32005)
    unstamped = root / "umbilicus.txt"
    unstamped.write_text("".join(f"{z}, 8, 8\n" for z in range(16)))
    error = expect_error(call, "umbilicus.import", {"path": str(unstamped), "replace": True}, -32602)
    assert error.data["param"] == "unitsPerVoxel", error.data
    replaced = call("umbilicus.import", {"path": str(unstamped), "replace": True, "unitsPerVoxel": 1})
    assert replaced["replaced"], replaced
    assert reopen()["umbilicus"]["uuids"] == [replaced["uuid"]]
    removed = call("umbilicus.remove")
    assert removed["removed"] == [replaced["uuid"]] and call("umbilicus.remove")["removed"] == []
    assert reopen()["umbilicus"]["uuids"] == []

    # Detaching the current volume leaves no stale selection; the volume's files stay.
    location = next(entry["location"] for entry in call("state.get")["project"]["entries"] if entry["kind"] == "volume"
                    and Path(entry["location"]).name == "scan")
    call("project.detach_entry", {"location": location})
    state = call("state.get")
    assert state["volume"] is None or state["volume"]["id"] != "scan", state["volume"]
    assert state["umbilicus"] is None or state["volume"] is not None
    assert volume.is_dir()
    assert all(entry["location"] != location for entry in reopen()["project"]["entries"])

    # The last store cannot be detached; a non-writable existing store is recorded read-only.
    call("stores.detach", {"storeId": extra["storeId"]})
    call("stores.detach", {"storeId": readonly["storeId"]})
    expect_error(call, "stores.detach", {"storeId": main}, -32005)
    Path(spare["path"]).chmod(0o444)
    try:
        locked = call("project.create", {"path": str(root / "locked"), "store": spare["path"]})
        assert json.loads(Path(locked["path"]).read_text())["stores"][0]["read_only"] is True
    finally:
        Path(spare["path"]).chmod(0o644)
    print("Stores, routes, Save As, resources and the umbilicus persist through the bridge", flush=True)


if __name__ == "__main__":
    main()
