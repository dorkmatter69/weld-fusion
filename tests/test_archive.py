"""Archive integrity and path-safety tests; standard library only."""

import io
import tarfile
import tempfile
import unittest
from pathlib import Path

from weldfusion.archive import inventory


def _archive(path, names):
    with tarfile.open(path, "w:gz") as tar:
        for name, body in names:
            info = tarfile.TarInfo(name)
            info.size = len(body)
            tar.addfile(info, io.BytesIO(body))


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_inventory_and_extract(self):
        archive = self.root / "small.tar.gz"
        _archive(archive, [("./data/a.csv", b"a,b\n1,2\n"), ("data/b.flac", b"fake")])
        report = inventory(archive, self.root / "staged")
        self.assertEqual(report["member_count"], 2)
        self.assertEqual(report["total_uncompressed_bytes"], 12)
        self.assertEqual((self.root / "staged/data/a.csv").read_bytes(), b"a,b\n1,2\n")
        self.assertEqual(len(report["members"][0]["sha256"]), 64)

    def test_archive_with_dot_root_directory(self):
        archive = self.root / "dot-root.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for name in ("./", "./s1/"):
                folder = tarfile.TarInfo(name)
                folder.type = tarfile.DIRTYPE
                tar.addfile(folder)
            item = tarfile.TarInfo("./s1/a.csv")
            item.size = 3
            tar.addfile(item, io.BytesIO(b"abc"))
        staged = self.root / "staged"
        report = inventory(archive, staged)
        self.assertEqual(report["member_count"], 1)
        self.assertEqual((staged / "s1/a.csv").read_bytes(), b"abc")

    def test_reject_unsafe_names(self):
        for i, name in enumerate(("../escape", "/absolute", "data/../escape", "data\\escape")):
            with self.subTest(name=name):
                archive = self.root / f"bad-{i}.tar.gz"
                _archive(archive, [(name, b"bad")])
                with self.assertRaisesRegex(ValueError, "unsafe"):
                    inventory(archive, self.root / "staged")

    def test_reject_symlink_parent(self):
        archive = self.root / "small.tar.gz"
        _archive(archive, [("data/a.csv", b"abc")])
        staged = self.root / "staged"
        staged.mkdir()
        (staged / "data").symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            inventory(archive, staged)

    def test_reject_duplicate_members(self):
        archive = self.root / "duplicate.tar.gz"
        _archive(archive, [("a.csv", b"one"), ("./a.csv", b"two")])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            inventory(archive)


if __name__ == "__main__":
    unittest.main()
