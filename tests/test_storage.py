import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photo_grouping import storage  # noqa: E402


class LocalStorageAdapterTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "originals"

    def tearDown(self):
        self._tmp.cleanup()

    def test_creates_root_dir_and_saves_bytes(self):
        adapter = storage.LocalStorageAdapter(self.root)
        backend, path = adapter.save_original(b"hello", "photo.jpg")
        self.assertEqual(backend, "local")
        self.assertEqual(Path(path).read_bytes(), b"hello")
        self.assertTrue(Path(path).is_relative_to(self.root))

    def test_does_not_overwrite_on_filename_collision(self):
        adapter = storage.LocalStorageAdapter(self.root)
        _, path1 = adapter.save_original(b"first", "IMG_0001.jpg")
        _, path2 = adapter.save_original(b"second", "IMG_0001.jpg")

        self.assertNotEqual(path1, path2)
        self.assertEqual(Path(path1).read_bytes(), b"first")
        self.assertEqual(Path(path2).read_bytes(), b"second")

    def test_icloud_adapter_reports_its_own_backend_name(self):
        adapter = storage.ICloudStorageAdapter(self.root)
        backend, _ = adapter.save_original(b"x", "a.jpg")
        self.assertEqual(backend, "icloud")


class GetAdapterTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_local_and_icloud_resolve(self):
        self.assertIsInstance(storage.get_adapter("local", self.root), storage.LocalStorageAdapter)
        self.assertIsInstance(storage.get_adapter("icloud", self.root), storage.ICloudStorageAdapter)

    def test_not_yet_built_backends_raise_not_implemented(self):
        for backend in ("google_drive", "dropbox", "s3"):
            with self.assertRaises(NotImplementedError):
                storage.get_adapter(backend, self.root)

    def test_unknown_backend_raises_value_error(self):
        with self.assertRaises(ValueError):
            storage.get_adapter("bogus", self.root)


if __name__ == "__main__":
    unittest.main()
