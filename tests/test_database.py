"""Unit tests for ApplyPilot database.py using unittest"""
import unittest
import tempfile
from pathlib import Path
from applypilot.database import get_connection, init_db, close_connection

class TestDatabase(unittest.TestCase):
    def test_init_db(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_file = Path(tmp_dir) / "test_applypilot.db"
            conn = init_db(db_file)
            self.assertIsNotNone(conn)

            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='jobs'")
            row = cursor.fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row[0], "jobs")
            close_connection(db_file)

    def test_thread_local_connection(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_file = Path(tmp_dir) / "test_applypilot.db"
            conn1 = get_connection(db_file)
            conn2 = get_connection(db_file)
            self.assertIs(conn1, conn2)
            close_connection(db_file)

if __name__ == "__main__":
    unittest.main()
