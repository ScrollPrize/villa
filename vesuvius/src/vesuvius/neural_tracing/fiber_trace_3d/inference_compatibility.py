"""Explicit, audited scheduling-only compatibility for published inference.

Never ignore implementation fingerprints wholesale. These exact source pairs
change GPU lifetime/manager control only; model, preprocessing, arithmetic,
runtime, and every other recorded hash must still match. New edits fail closed.
Existing cache metadata is not rewritten or relabelled.
"""
SCHEDULING_ONLY = {
    "lasagna.demand_predict3d": (
        "51f0653ae450c523bf0c8ac9e3ab18553852941a09ee62f700cbdd2281bf0535",
        "d5802c52486d49cafbdab225c02ece371e46066b2dba0bbb4c2db664b07e13f2"),
    "runner": (
        "8a46be0e6f5e1aefb9956320d3fe19c114fe316c65599e0001e0d25887b9bd02",
        "90dfed5b9907d869a0ac242b4aac870aeb8ff57047b0c01f8b2f91dcdb4ecf64"),
}


def compatible_implementation(recorded, current):
    if recorded == current:
        return True
    normalized = dict(recorded)
    for key, (old, new) in SCHEDULING_ONLY.items():
        if normalized.get(key) == old and current.get(key) == new:
            normalized[key] = new
    if normalized != current:
        return False
    print("inference cache: accepted audited scheduling-only GPU lifetime update; numerical identity unchanged", flush=True)
    return True
