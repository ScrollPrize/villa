import dbm
import importlib.util
import pickle
import sqlite3
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import tracks


def line_track(length, *, z=10, y=10, axis=2):
    points = np.zeros((int(length) + 1, 3), dtype=np.int32)
    points[:, 0] = z
    points[:, 1] = y
    points[:, axis] = np.arange(int(length) + 1, dtype=np.int32)
    return points


def write_sqlite_dbm(path, items):
    # Same single-table layout as CPython's Lib/dbm/sqlite3.py, written with
    # plain sqlite3 so the fixture also builds on Python < 3.13.
    connection = sqlite3.connect(path)
    connection.execute('CREATE TABLE IF NOT EXISTS Dict (key BLOB UNIQUE NOT NULL, value BLOB NOT NULL)')
    connection.executemany('REPLACE INTO Dict (key, value) VALUES (CAST(? AS BLOB), CAST(? AS BLOB))', items)
    connection.commit()
    connection.close()


class SQLiteTrackDBMTests(unittest.TestCase):
    def make_db(self, root):
        path = Path(root) / 'tracks.dbm'
        horizontal = line_track(20, z=10, y=10, axis=2)
        vertical = line_track(20, z=10, y=0, axis=1)
        write_sqlite_dbm(str(path), [
            (b'h:0', pickle.dumps([horizontal])),
            (b'vy:0', pickle.dumps([vertical])),
        ])
        return path

    def test_sqlite_store_loads_without_dbm_sqlite3(self):
        # Simulate an interpreter without dbm.sqlite3 (Python < 3.13): dbm can
        # neither identify nor open the published SQLite-backed store.
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_db(temporary)
            with (mock.patch.object(tracks.dbm, 'whichdb', return_value=''),
                  mock.patch.object(tracks.dbm, 'open', side_effect=dbm.error[0]('db type could not be determined'))):
                self.assertEqual(tracks.normalize_tracks_dbm_path(path), str(path))
                with tracks.open_tracks_dbm(path) as database:
                    self.assertEqual(sorted(database.keys()), [b'h:0', b'vy:0'])
                    self.assertEqual(len(pickle.loads(database[b'h:0'])), 1)
                loaded = tracks.load_tracks_from_dbm(path, show_progress=False)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(sorted(len(track) for track in loaded), [21, 21])

    @unittest.skipUnless(importlib.util.find_spec('dbm.sqlite3'), 'needs dbm.sqlite3 (Python >= 3.13)')
    def test_native_dbm_path_is_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_db(temporary)
            self.assertEqual(dbm.whichdb(str(path)), 'dbm.sqlite3')
            with tracks.open_tracks_dbm(path) as database:
                self.assertNotIsInstance(database, tracks._SQLiteTrackDBM)
                self.assertEqual(sorted(database.keys()), [b'h:0', b'vy:0'])


if __name__ == '__main__':
    unittest.main()
