-- Frozen schema v1 fixture, before provenance/patch-scale migration.
PRAGMA application_id=1447249746;
PRAGMA user_version=1;

CREATE TABLE objects(
  id INTEGER PRIMARY KEY,
  uuid BLOB NOT NULL UNIQUE,
  kind TEXT NOT NULL CHECK(length(kind)>0),
  name TEXT NOT NULL DEFAULT '',
  generation INTEGER NOT NULL DEFAULT 1,
  created_by TEXT NOT NULL, created_at TEXT NOT NULL, created_host TEXT NOT NULL,
  changed_by TEXT NOT NULL, changed_at TEXT NOT NULL, changed_host TEXT NOT NULL,
  approval_state TEXT NOT NULL DEFAULT 'pending' CHECK(approval_state IN ('pending','approved','rejected')),
  approved_by TEXT, approved_at TEXT, approved_host TEXT, approved_generation INTEGER,
  format_version INTEGER NOT NULL DEFAULT 0,
  storage_kind TEXT NOT NULL DEFAULT '', storage_ref TEXT NOT NULL DEFAULT '',
  data_generation INTEGER NOT NULL DEFAULT 0, data_checksum TEXT,
  meta TEXT NOT NULL DEFAULT '{}' CHECK(json_valid(meta) AND json_type(meta)='object')
);
CREATE INDEX objects_kind ON objects(kind);
CREATE INDEX objects_approval ON objects(approval_state);
CREATE TABLE fibers(
  object_id INTEGER PRIMARY KEY REFERENCES objects(id) ON DELETE CASCADE,
  family TEXT NOT NULL DEFAULT '', manual_family INTEGER NOT NULL DEFAULT 0,
  control_point_count INTEGER NOT NULL DEFAULT 0, line_point_count INTEGER NOT NULL DEFAULT 0,
  length REAL NOT NULL DEFAULT 0, width REAL NOT NULL DEFAULT 0, width_gap_fraction REAL NOT NULL DEFAULT 0.2,
  optimization_mode TEXT NOT NULL DEFAULT '',
  bbox_min_x REAL, bbox_min_y REAL, bbox_min_z REAL, bbox_max_x REAL, bbox_max_y REAL, bbox_max_z REAL
);
CREATE TABLE patches(
  object_id INTEGER PRIMARY KEY REFERENCES objects(id) ON DELETE CASCADE,
  grid_width INTEGER NOT NULL DEFAULT 0, grid_height INTEGER NOT NULL DEFAULT 0, scale REAL NOT NULL DEFAULT 0,
  bbox_min_x REAL, bbox_min_y REAL, bbox_min_z REAL, bbox_max_x REAL, bbox_max_y REAL, bbox_max_z REAL
);
CREATE TABLE tracks(
  object_id INTEGER PRIMARY KEY REFERENCES objects(id) ON DELETE CASCADE,
  dataset_ref TEXT NOT NULL DEFAULT '', external_key TEXT NOT NULL DEFAULT '',
  family TEXT NOT NULL DEFAULT '', point_count INTEGER NOT NULL DEFAULT 0, length REAL NOT NULL DEFAULT 0,
  bbox_min_x REAL, bbox_min_y REAL, bbox_min_z REAL, bbox_max_x REAL, bbox_max_y REAL, bbox_max_z REAL
);
CREATE UNIQUE INDEX tracks_identity ON tracks(dataset_ref, external_key) WHERE external_key<>'';
CREATE TABLE object_tags(
  object_id INTEGER NOT NULL REFERENCES objects(id) ON DELETE CASCADE,
  tag TEXT NOT NULL, PRIMARY KEY(object_id, tag)
);
CREATE INDEX object_tags_tag ON object_tags(tag);
CREATE TABLE edges(
  id INTEGER PRIMARY KEY,
  uuid BLOB NOT NULL UNIQUE,
  a_id INTEGER NOT NULL REFERENCES objects(id) ON DELETE RESTRICT,
  b_id INTEGER NOT NULL REFERENCES objects(id) ON DELETE RESTRICT,
  kind TEXT NOT NULL CHECK(length(kind)>0),
  generation INTEGER NOT NULL DEFAULT 1,
  created_by TEXT NOT NULL, created_at TEXT NOT NULL, created_host TEXT NOT NULL,
  changed_by TEXT NOT NULL, changed_at TEXT NOT NULL, changed_host TEXT NOT NULL,
  approval_state TEXT NOT NULL DEFAULT 'pending' CHECK(approval_state IN ('pending','approved','rejected')),
  approved_by TEXT, approved_at TEXT, approved_host TEXT, approved_generation INTEGER,
  a_anchor_index INTEGER, a_anchor_x REAL, a_anchor_y REAL, a_anchor_z REAL,
  a_dir_x REAL, a_dir_y REAL, a_dir_z REAL,
  b_anchor_index INTEGER, b_anchor_x REAL, b_anchor_y REAL, b_anchor_z REAL,
  b_dir_x REAL, b_dir_y REAL, b_dir_z REAL,
  meta TEXT NOT NULL DEFAULT '{}' CHECK(json_valid(meta) AND json_type(meta)='object'),
  CHECK(a_id < b_id)
);
CREATE INDEX edges_a ON edges(a_id, kind);
CREATE INDEX edges_b ON edges(b_id, kind);
CREATE UNIQUE INDEX edges_unique_unanchored ON edges(kind, a_id, b_id)
  WHERE a_anchor_index IS NULL AND b_anchor_index IS NULL;
CREATE UNIQUE INDEX edges_unique_anchored ON edges(kind, a_id, a_anchor_index, b_id, b_anchor_index)
  WHERE a_anchor_index IS NOT NULL AND b_anchor_index IS NOT NULL;
CREATE TABLE edge_tags(
  edge_id INTEGER NOT NULL REFERENCES edges(id) ON DELETE CASCADE,
  tag TEXT NOT NULL, PRIMARY KEY(edge_id, tag)
);
CREATE INDEX edge_tags_tag ON edge_tags(tag);
CREATE TABLE journal(
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  record_uuid BLOB NOT NULL, generation INTEGER NOT NULL,
  author TEXT NOT NULL, hostname TEXT NOT NULL, timestamp TEXT NOT NULL,
  change_kind TEXT NOT NULL
);
CREATE INDEX journal_record ON journal(record_uuid, seq);
