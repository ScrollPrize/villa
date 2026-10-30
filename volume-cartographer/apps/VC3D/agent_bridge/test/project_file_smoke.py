#!/usr/bin/env python3
"""Exercise standalone projects against the real VC3D bridge, without network data."""
from __future__ import annotations

import argparse
import json
import tempfile
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


if __name__ == "__main__":
    main()
