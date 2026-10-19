#pragma once
#include "internal/Database.hpp"
namespace vc::objects { namespace {
const char* kObjectSchema = R"SQL(CREATE TABLE objects(
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
  scale_y REAL NOT NULL DEFAULT 0,
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
CREATE TABLE journal(
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  record_uuid BLOB NOT NULL, generation INTEGER NOT NULL,
  author TEXT NOT NULL, hostname TEXT NOT NULL, timestamp TEXT NOT NULL,
  change_kind TEXT NOT NULL,
  approval_state TEXT, approved_by TEXT, approved_at TEXT, approved_host TEXT, approved_generation INTEGER
);
CREATE INDEX journal_record ON journal(record_uuid, seq);
)SQL";
const char* kEdgeSchema = R"SQL(CREATE TABLE edges(
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
)SQL";
const char* kObjectGeometrySchema = R"SQL(CREATE TABLE geometry_chunks(
  hash TEXT PRIMARY KEY CHECK(length(hash)=64),
  data BLOB NOT NULL CHECK(length(data)>0 AND length(data)<=1048576)
);
CREATE TABLE geometry_revisions(
  uuid BLOB PRIMARY KEY CHECK(length(uuid)=16),
  object_uuid BLOB NOT NULL REFERENCES objects(uuid) ON DELETE RESTRICT,
  parent_uuid BLOB,
  content_hash TEXT NOT NULL CHECK(length(content_hash)=64),
  manifest TEXT NOT NULL CHECK(json_valid(manifest) AND json_type(manifest)='object'),
  object_generation INTEGER NOT NULL,
  created_by TEXT NOT NULL, created_at TEXT NOT NULL, created_host TEXT NOT NULL,
  UNIQUE(object_uuid,uuid),
  FOREIGN KEY(object_uuid,parent_uuid) REFERENCES geometry_revisions(object_uuid,uuid)
);
CREATE INDEX geometry_history ON geometry_revisions(object_uuid,object_generation);
CREATE TABLE geometry_heads(
  object_uuid BLOB PRIMARY KEY REFERENCES objects(uuid) ON DELETE RESTRICT,
  revision_uuid BLOB NOT NULL,
  FOREIGN KEY(object_uuid,revision_uuid) REFERENCES geometry_revisions(object_uuid,uuid)
);
CREATE TABLE geometry_revision_chunks(
  revision_uuid BLOB NOT NULL REFERENCES geometry_revisions(uuid),
  array_name TEXT NOT NULL, chunk_index INTEGER NOT NULL CHECK(chunk_index>=0),
  hash TEXT NOT NULL REFERENCES geometry_chunks(hash),
  PRIMARY KEY(revision_uuid,array_name,chunk_index)
);
CREATE TRIGGER geometry_chunks_immutable_update BEFORE UPDATE ON geometry_chunks
  BEGIN SELECT RAISE(ABORT,'geometry chunks are immutable'); END;
CREATE TRIGGER geometry_chunks_immutable_delete BEFORE DELETE ON geometry_chunks
  BEGIN SELECT RAISE(ABORT,'geometry chunks are retained'); END;
CREATE TRIGGER geometry_revisions_immutable_update BEFORE UPDATE ON geometry_revisions
  BEGIN SELECT RAISE(ABORT,'geometry revisions are immutable'); END;
CREATE TRIGGER geometry_revisions_immutable_delete BEFORE DELETE ON geometry_revisions
  BEGIN SELECT RAISE(ABORT,'geometry revisions are retained'); END;
CREATE TRIGGER geometry_revision_chunks_immutable_update BEFORE UPDATE ON geometry_revision_chunks
  BEGIN SELECT RAISE(ABORT,'geometry chunk references are immutable'); END;
CREATE TRIGGER geometry_revision_chunks_immutable_delete BEFORE DELETE ON geometry_revision_chunks
  BEGIN SELECT RAISE(ABORT,'geometry chunk references are retained'); END;
)SQL";
const char* kEdgeGeometrySchema = R"SQL(
CREATE TABLE edge_geometry_bindings(
  edge_id INTEGER PRIMARY KEY REFERENCES edges(id) ON DELETE CASCADE,
  a_revision BLOB, b_revision BLOB
);
)SQL";
void createSchema(sqlite3* db) {
    exec(db, kObjectSchema);
    exec(db, kEdgeSchema);
    exec(db, kEdgeGeometrySchema);
    exec(db, kObjectGeometrySchema);
    exec(db, "CREATE TABLE store_identity(singleton INTEGER PRIMARY KEY CHECK(singleton=1), uuid BLOB NOT NULL UNIQUE CHECK(length(uuid)=16));");
    Statement id(db, "INSERT INTO store_identity VALUES(1,?)");
    id.bindBlob(1, parseUuid(util::newUuid())); id.stepDone();
    exec(db, "CREATE TRIGGER store_identity_immutable_update BEFORE UPDATE ON store_identity "
             "BEGIN SELECT RAISE(ABORT,'store identity is immutable'); END;"
             "CREATE TRIGGER store_identity_immutable_delete BEFORE DELETE ON store_identity "
             "BEGIN SELECT RAISE(ABORT,'store identity is retained'); END;");

}

} }
