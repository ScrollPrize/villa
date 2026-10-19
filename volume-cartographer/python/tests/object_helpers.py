"""Explicit caller decisions used in object-store tests."""
def edge_choice(db, edge_uuid, action, **anchors):
    edge = db.edge(edge_uuid)
    revisions = [db.current_geometry(edge[side]) for side in ("a", "b")]
    return dict(edge_uuid=edge_uuid, expected_generation=edge["generation"], action=action,
                a_revision=revisions[0]["uuid"] if revisions[0] else None,
                b_revision=revisions[1]["uuid"] if revisions[1] else None, **anchors)


def canonical_edge(edge):
    edge = dict(edge)
    if edge["b"] < edge["a"]:
        edge["a"], edge["b"] = edge["b"], edge["a"]
        edge["a_anchor"], edge["b_anchor"] = edge["b_anchor"], edge["a_anchor"]
    return edge
