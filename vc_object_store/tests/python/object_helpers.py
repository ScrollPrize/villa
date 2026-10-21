def edge_choice(db, edge_uuid, action, **anchors):
    edge = db.edge(edge_uuid)
    revisions = [db.current_geometry(edge[side]) for side in ("a", "b")]
    return dict(edge_uuid=edge_uuid, expected_generation=edge["generation"], action=action,
                a_revision=revisions[0]["uuid"] if revisions[0] else None,
                b_revision=revisions[1]["uuid"] if revisions[1] else None, **anchors)
